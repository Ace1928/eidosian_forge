from pythran.analyses import LocalNodeDeclarations, GlobalDeclarations, Scope
from pythran.analyses import YieldPoints, IsAssigned, ASTMatcher, AST_any
from pythran.analyses import RangeValues, PureExpressions, Dependencies
from pythran.analyses import Immediates, Ancestors, StrictAliases
from pythran.config import cfg
from pythran.cxxgen import Template, Include, Namespace, CompilationUnit
from pythran.cxxgen import Statement, Block, AnnotatedStatement, Typedef, Label
from pythran.cxxgen import Value, FunctionDeclaration, EmptyStatement, Nop
from pythran.cxxgen import FunctionBody, Line, ReturnStatement, Struct, Assign
from pythran.cxxgen import For, While, TryExcept, ExceptHandler, If, AutoFor
from pythran.cxxgen import StatementWithComments
from pythran.openmp import OMPDirective
from pythran.passmanager import Backend
from pythran.syntax import PythranSyntaxError
from pythran.tables import operator_to_lambda, update_operator_to_lambda
from pythran.tables import pythran_ward, attributes as attributes_table
from pythran.types.conversion import PYTYPE_TO_CTYPE_TABLE, TYPE_TO_SUFFIX
from pythran.types.types import Types
from pythran.utils import attr_to_path, pushpop, cxxid, isstr, isnum
from pythran.utils import isextslice, ispowi, quote_cxxstring
from pythran import metadata, unparse
from math import isnan, isinf
import gast as ast
from functools import reduce
import io
class CxxFunction(ast.NodeVisitor):
    """
    Attributes
    ----------
    ldecls : {str}
        set of local declarations.
    break_handler : [str]
        It contains flags for goto statements to jump on break in case of
        orelse statement in loop. None means there are no orelse statement so
        no jump are requiered.
        (else in loop means : don't execute if loop is terminated with a break)
    """

    def __init__(self, parent):
        """ Basic initialiser gathering analysis informations. """
        self.parent = parent
        self.break_handlers = []
        self.used_break = set()
        self.ldecls = None
        self.openmp_deps = set()
        if not (cfg.getboolean('backend', 'annotate') and self.passmanager.code):
            self.add_line_info = self.skip_line_info
        else:
            self.lines = self.passmanager.code.split('\n')

    def __getattr__(self, attr):
        return getattr(self.parent, attr)

    def process_locals(self, node, node_visited, *skipped):
        """
        Declare variable local to node and insert declaration before.

        Not possible for function yielding values.
        """
        local_vars = self.scope[node].difference(skipped)
        local_vars = local_vars.difference(self.openmp_deps)
        if not local_vars:
            return node_visited
        locals_visited = []
        for varname in local_vars:
            vartype = self.typeof(varname)
            decl = Statement('{} {}'.format(vartype, varname))
            locals_visited.append(decl)
        self.ldecls.difference_update(local_vars)
        return Block(locals_visited + [node_visited])

    def visit_OMPDirective(self, node):
        self.openmp_deps.update((d.id for d in node.private_deps))
        self.openmp_deps.update((d.id for d in node.shared_deps))

    def add_line_info(self, node, cxx_node):
        if not isinstance(node, ast.stmt):
            return cxx_node
        line = self.lines[node.lineno - 1].rstrip()
        if isinstance(node, ast.FunctionDef):
            head, tail = cxx_node
            return (head, [StatementWithComments(t, line) for t in tail])
        return StatementWithComments(cxx_node, line)

    def skip_line_info(self, node, cxx_node):
        return cxx_node

    def visit(self, node):
        metadata.visit(self, node)
        result = super(CxxFunction, self).visit(node)
        return self.add_line_info(node, result)

    def process_omp_attachements(self, node, stmt, index=None):
        """
        Add OpenMP pragma on the correct stmt in the correct order.

        stmt may be a list. On this case, index have to be specify to add
        OpenMP on the correct statement.
        """
        omp_directives = metadata.get(node, OMPDirective)
        if omp_directives:
            directives = list()
            for directive in omp_directives:
                directive.deps = [self.visit(dep) for dep in directive.deps]
                directives.append(directive)
            if index is None:
                stmt = AnnotatedStatement(stmt, directives)
            else:
                stmt[index] = AnnotatedStatement(stmt[index], directives)
        return stmt

    def typeof(self, node):
        if isinstance(node, str):
            return self.typeof(self.local_names[node])
        elif isinstance(node, ast.AST):
            return self.lctx(self.types[node])
        else:
            return self.lctx(node)

    def prepare_functiondef_context(self, node):
        fargs = node.args.args
        formal_args = [cxxid(arg.id) for arg in fargs]
        formal_types = ['argument_type' + str(i) for i in range(len(fargs))]
        local_decls = set(self.gather(LocalNodeDeclarations, node))
        self.local_names = {sym.id: sym for sym in local_decls}
        self.local_names.update({arg.id: arg for arg in fargs})
        self.lctx = CachedTypeVisitor()
        self.ldecls = {n.id for n in local_decls}
        body = [self.visit(stmt) for stmt in node.body]
        return (body, formal_types, formal_args)

    def prepare_types(self, node):
        dflt_argv = [None] * (len(node.args.args) - len(node.args.defaults)) + [self.visit(n) for n in node.args.defaults]
        dflt_argt = [None] * (len(node.args.args) - len(node.args.defaults)) + [self.types[n].sgenerate() for n in node.args.defaults]
        result_type = self.types[node][0]
        callable_type = Typedef(Value('void', 'callable'))
        pure_type = Typedef(Value('void', 'pure')) if node in self.pure_expressions else EmptyStatement()
        return (dflt_argv, dflt_argt, result_type, callable_type, pure_type)

    def visit_FunctionDef(self, node):
        self.fname = cxxid(node.name)
        tmp = self.prepare_functiondef_context(node)
        operator_body, formal_types, formal_args = tmp
        tmp = self.prepare_types(node)
        dflt_argv, dflt_argt, result_type, callable_type, pure_type = tmp
        fscope = 'type{0}::'.format('<{0}>'.format(', '.join(formal_types)) if formal_types else '')
        ffscope = '{0}::{1}'.format(self.fname, fscope)
        operator_declaration = [templatize(make_const_function_declaration(self, node, 'typename {0}result_type'.format(fscope), 'operator()', formal_types, formal_args, dflt_argv), formal_types, dflt_argt), EmptyStatement()]
        operator_signature = make_const_function_declaration(self, node, 'typename {0}result_type'.format(ffscope), '{0}::operator()'.format(self.fname), formal_types, formal_args)
        ctx = CachedTypeVisitor(self.lctx)
        operator_local_declarations = [Statement('{0} {1}'.format(ctx(self.types[self.local_names[k]]), cxxid(k))) for k in self.ldecls]
        dependent_typedefs = ctx.typedefs()
        operator_definition = FunctionBody(templatize(operator_signature, formal_types), Block(dependent_typedefs + operator_local_declarations + operator_body))
        ctx = CachedTypeVisitor()
        extra_typedefs = [Typedef(Value(ctx(t), t.name)) for t in self.types[node][1]] + [Typedef(Value(ctx(result_type), 'result_type'))]
        extra_typedefs = ctx.typedefs() + extra_typedefs
        return_declaration = [templatize(Struct('type', extra_typedefs), formal_types, dflt_argt)]
        topstruct = Struct(self.fname, [callable_type, pure_type] + return_declaration + operator_declaration)
        topstruct = self.process_omp_attachements(node, topstruct)
        return ([topstruct], [operator_definition])

    def visit_Return(self, node):
        value = self.visit(node.value)
        if metadata.get(node, metadata.StaticReturn):
            rtype = 'typename {}::type::result_type'.format(self.fname)
            stmt = Block([Assign('static %s tmp_global' % rtype, value), ReturnStatement('tmp_global')])
        else:
            stmt = ReturnStatement(value)
        return self.process_omp_attachements(node, stmt)

    def visit_Delete(self, _):
        return Nop()

    def visit_Assign(self, node):
        """
        Create Assign node for final Cxx representation.

        It tries to handle multi assignment like:

        >> a = b = c = 2

        If only one local variable is assigned, typing is added:

        >> int a = 2;

        TODO: Handle case of multi-assignement for some local variables.

        Finally, process OpenMP clause like #pragma omp atomic
        """
        if not all((isinstance(n, (ast.Name, ast.Subscript)) for n in node.targets)):
            raise PythranSyntaxError('Must assign to an identifier or a subscript', node)
        value = self.visit(node.value)
        targets = [self.visit(t) for t in node.targets]
        alltargets = '= '.join(targets)
        islocal = len(targets) == 1 and isinstance(node.targets[0], ast.Name) and (node.targets[0].id in self.scope[node]) and (node.targets[0].id not in self.openmp_deps)
        if islocal:
            self.ldecls.difference_update((t.id for t in node.targets))
            if self.types[node.targets[0]].iscombined():
                alltargets = '{} {}'.format(self.typeof(node.targets[0]), alltargets)
            elif isinstance(self.types[node.targets[0]], self.types.builder.Assignable):
                alltargets = '{} {}'.format(self.types.builder.AssignableNoEscape(self.types.builder.NamedType('decltype({})'.format(value))).sgenerate(), alltargets)
            else:
                assert isinstance(self.types[node.targets[0]], self.types.builder.Lazy)
                alltargets = '{} {}'.format(self.types.builder.Lazy(self.types.builder.NamedType('decltype({})'.format(value))).sgenerate(), alltargets)
        stmt = Assign(alltargets, value)
        return self.process_omp_attachements(node, stmt)

    def visit_AugAssign(self, node):
        value = self.visit(node.value)
        target = self.visit(node.target)
        op = update_operator_to_lambda[type(node.op)]
        stmt = Statement(op(target, value)[1:-1])
        return self.process_omp_attachements(node, stmt)

    def visit_Print(self, node):
        values = [self.visit(n) for n in node.values]
        stmt = Statement('pythonic::builtins::print{0}({1})'.format('' if node.nl else '_nonl', ', '.join(values)))
        return self.process_omp_attachements(node, stmt)

    def is_in_collapse(self, loop, node):
        for ancestor in reversed(self.ancestors[loop]):
            if not isinstance(ancestor, ast.For):
                return False
            for directive in metadata.get(ancestor, OMPDirective):
                if 'collapse' in directive.s:
                    if node not in self.pure_expressions:
                        raise PythranSyntaxError('not pure expression used as loop target inside a collapse clause', loop)
                    return True
        assert False, 'unreachable state'

    def gen_for(self, node, target, local_iter, local_iter_decl, loop_body):
        """
        Create For representation on iterator for Cxx generation.

        Examples
        --------
        >> "omp parallel for"
        >> for i in range(10):
        >>     ... do things ...

        Becomes

        >> "omp parallel for shared(__iterX)"
        >> for(decltype(__iterX)::iterator __targetX = __iterX.begin();
               __targetX < __iterX.end(); ++__targetX)
        >>         auto&& i = *__targetX;
        >>     ... do things ...

        It the case of not local variable, typing for `i` disappear and typing
        is removed for iterator in case of yields statement in function.
        """
        local_target = '__target{0}'.format(id(node))
        local_target_decl = self.typeof(self.types.builder.IteratorOfType(local_iter_decl))
        islocal = node.target.id not in self.openmp_deps and node.target.id in self.scope[node] and (not hasattr(self, 'yields'))
        if islocal:
            local_type = 'auto&&'
            self.ldecls.remove(node.target.id)
        else:
            local_type = ''
        loop_body_prelude = Statement('{} {}= *{}'.format(local_type, target, local_target))
        assign = self.make_assign(local_target_decl, local_target, local_iter)
        loop = For('{}.begin()'.format(assign), '{0} < {1}.end()'.format(local_target, local_iter), '++{0}'.format(local_target), Block([loop_body_prelude, loop_body]))
        return [self.process_omp_attachements(node, loop)]

    def handle_real_loop_comparison(self, args, target, upper_bound):
        """
        Handle comparison for real loops.

        Add the correct comparison operator if possible.
        """
        if len(args) <= 2:
            order = 1
        elif isnum(args[2]):
            order = -1 + 2 * (int(args[2].value) > 0)
        elif isnum(args[1]) and isnum(args[0]):
            order = -1 + 2 * (int(args[1].value) > int(args[0].value))
        else:
            order = 0
        comparison = '{} < {}' if order == 1 else '{} > {}'
        comparison = comparison.format(target, upper_bound)
        return comparison

    def gen_c_for(self, node, local_iter, loop_body):
        """
        Create C For representation for Cxx generation.

        Examples
        --------
        >> for i in range(10):
        >>     ... do things ...

        Becomes

        >> for(long i = 0, __targetX = 10; i < __targetX; i += 1)
        >>     ... do things ...

        Or

        >> for i in range(10, 0, -1):
        >>     ... do things ...

        Becomes

        >> for(long i = 10, __targetX = 0; i > __targetX; i += -1)
        >>     ... do things ...


        It the case of not local variable, typing for `i` disappear
        """
        args = node.iter.args
        step = '1L' if len(args) <= 2 else self.visit(args[2])
        if len(args) == 1:
            lower_bound = '0L'
            upper_arg = 0
        else:
            lower_bound = self.visit(args[0])
            upper_arg = 1
        upper_type = iter_type = 'long '
        upper_value = self.visit(args[upper_arg])
        if self.is_in_collapse(node, args[upper_arg]):
            upper_bound = upper_value
        else:
            upper_bound = '__target{0}'.format(id(node))
        islocal = node.target.id not in self.openmp_deps and node.target.id in self.scope[node] and (not hasattr(self, 'yields'))
        if islocal:
            loop = list()
            self.ldecls.remove(node.target.id)
        else:
            iter_type = ''
            if node.target.id in self.scope[node]:
                loop = []
            else:
                loop = [If('{} == {}'.format(local_iter, upper_bound), Statement('{} -= {}'.format(local_iter, step)))]
        comparison = self.handle_real_loop_comparison(args, local_iter, upper_bound)
        forloop = For('{0} {1}={2}'.format(iter_type, local_iter, lower_bound), comparison, '{0} += {1}'.format(local_iter, step), loop_body)
        loop.insert(0, self.process_omp_attachements(node, forloop))
        if upper_bound is upper_value:
            header = []
        else:
            assgnt = self.make_assign(upper_type, upper_bound, upper_value)
            header = [Statement(assgnt)]
        return (header, loop)

    def handle_omp_for(self, node, local_iter):
        """
        Fix OpenMP directives on For loops.

        Add the target as private variable as a new variable may have been
        introduce to handle cxx iterator.

        Also, add the iterator as shared variable as all 'parallel for chunck'
        have to use the same iterator.
        """
        for directive in metadata.get(node, OMPDirective):
            if any((key in directive.s for key in (' parallel ', ' task '))):
                directive.s += ' shared({})'
                directive.deps.append(ast.Name(local_iter, ast.Load(), None, None))
                directive.shared_deps.append(directive.deps[-1])
            target = node.target
            assert isinstance(target, ast.Name)
            hasfor = 'for' in directive.s
            nodefault = 'default' not in directive.s
            noindexref = all((isinstance(x, ast.Name) and x.id != target.id for x in directive.deps))
            if hasfor and nodefault and noindexref and (target.id not in self.scope[node]):
                directive.s += ' private({})'
                directive.deps.append(ast.Name(target.id, ast.Load(), None, None))
                directive.private_deps.append(directive.deps[-1])

    def can_use_autofor(self, node):
        """
        Check if given for Node can use autoFor syntax.

        To use auto_for:
            - iterator should have local scope
            - yield should not be use
            - OpenMP pragma should not be use

        TODO : Yield should block only if it is use in the for loop, not in the
               whole function.
        """
        auto_for = isinstance(node.target, ast.Name) and node.target.id in self.scope[node] and (node.target.id not in self.openmp_deps)
        auto_for &= not metadata.get(node, OMPDirective)
        return auto_for

    def can_use_c_for(self, node):
        """
        Check if a for loop can use classic C syntax.

        To use C syntax:
            - target should not be assign in the loop
            - range should be use as iterator
            - order have to be known at compile time
        """
        assert isinstance(node.target, ast.Name)
        pattern_range = ast.Call(func=ast.Attribute(value=ast.Name('builtins', ast.Load(), None, None), attr='range', ctx=ast.Load()), args=AST_any(), keywords=[])
        is_assigned = set()
        for stmt in node.body:
            is_assigned.update({n.id for n in self.gather(IsAssigned, stmt)})
        match = ASTMatcher(pattern_range).match(node.iter)
        if not match:
            return False
        if node.target.id in is_assigned:
            return False
        args = node.iter.args
        if len(args) < 3:
            return True
        if isnum(args[2]):
            return True
        return False

    def make_assign(self, local_iter_decl, local_iter, iterable):
        return '{0} {1} = {2}'.format(local_iter_decl, local_iter, iterable)

    def is_user_function(self, func):
        aliases = self.strict_aliases[func]
        if not aliases:
            return False
        for alias in aliases:
            if not isinstance(alias, ast.FunctionDef):
                return False
            if self.gather(YieldPoints, alias):
                return False
        return True

    @cxx_loop
    def visit_For(self, node):
        """
        Create For representation for Cxx generation.

        Examples
        --------
        >> for i in range(10):
        >>     ... work ...

        Becomes

        >> typename returnable<decltype(builtins.range(10))>::type __iterX
           = builtins.range(10);
        >> ... possible container size reservation ...
        >> for (auto&& i: __iterX)
        >>     ... the work ...

        This function also handle assignment for local variables.

        We can notice that three kind of loop are possible:
        - Normal for loop on iterator
        - Autofor loop.
        - Normal for loop using integer variable iteration
        Kind of loop used depend on OpenMP, yield use and variable scope.
        """
        if not isinstance(node.target, ast.Name):
            raise PythranSyntaxError('Using something other than an identifier as loop target', node.target)
        target = self.visit(node.target)
        loop_body = Block([self.visit(stmt) for stmt in node.body])
        loop_body = self.process_locals(node, loop_body, node.target.id)
        iterable = self.visit(node.iter)
        if self.can_use_c_for(node):
            header, loop = self.gen_c_for(node, target, loop_body)
        elif self.can_use_autofor(node):
            header = []
            self.ldecls.remove(node.target.id)
            autofor = AutoFor(target, iterable, loop_body)
            loop = [self.process_omp_attachements(node, autofor)]
        else:
            local_iter = '__iter{0}'.format(id(node))
            local_iter_decl = self.types.builder.Assignable(self.types[node.iter])
            self.handle_omp_for(node, local_iter)
            asgnt = self.make_assign(self.typeof(local_iter_decl), local_iter, iterable)
            header = [Statement(asgnt)]
            loop = self.gen_for(node, target, local_iter, local_iter_decl, loop_body)
        for comp in metadata.get(node, metadata.Comprehension):
            header.append(Statement('pythonic::utils::reserve({0},{1})'.format(comp.target, iterable)))
        return Block(header + loop)

    @cxx_loop
    def visit_While(self, node):
        """
        Create While node for Cxx generation.

        It is a cxx_loop to handle else clause.
        """
        test = self.visit(node.test)
        body = [self.visit(n) for n in node.body]
        stmt = While(test, Block(body))
        return self.process_omp_attachements(node, stmt)

    def visit_Try(self, node):
        body = [self.visit(n) for n in node.body]
        except_ = list()
        for n in node.handlers:
            except_.extend(self.visit(n))
        return TryExcept(Block(body), except_)

    def visit_ExceptHandler(self, node):
        name = self.visit(node.name) if node.name else None
        body = [self.visit(m) for m in node.body]
        if isinstance(node.type, ast.Tuple):
            return [ExceptHandler(p.attr, Block(body), name) for p in node.type.elts]
        else:
            return [ExceptHandler(node.type and node.type.attr, Block(body), name)]

    def visit_If(self, node):
        test = self.visit(node.test)
        body = [self.visit(n) for n in node.body]
        orelse = [self.visit(n) for n in node.orelse]
        if isnum(node.test) and node.test.value == 1:
            stmt = Block(body)
        else:
            stmt = If(test, Block(body), Block(orelse) if orelse else None)
        return self.process_locals(node, self.process_omp_attachements(node, stmt))

    def visit_Raise(self, node):
        exc = node.exc and self.visit(node.exc)
        return Statement('throw {0}'.format(exc or ''))

    def visit_Assert(self, node):
        params = [self.visit(node.test), node.msg and self.visit(node.msg)]
        sparams = ', '.join((_f for _f in params if _f))
        return Statement('pythonic::pythran_assert({0})'.format(sparams))

    def visit_Import(self, _):
        return Nop()

    def visit_ImportFrom(self, _):
        assert False, 'should be filtered out by the expand_import pass'

    def visit_Expr(self, node):
        stmt = Statement(self.visit(node.value))
        return self.process_locals(node, self.process_omp_attachements(node, stmt))

    def visit_Pass(self, node):
        stmt = EmptyStatement()
        return self.process_omp_attachements(node, stmt)

    def visit_Break(self, _):
        """
        Generate break statement in most case and goto for orelse clause.

        See Also : cxx_loop
        """
        if self.break_handlers and self.break_handlers[-1]:
            self.used_break.add(self.break_handlers[-1])
            return Statement('goto {0}'.format(self.break_handlers[-1]))
        else:
            return Statement('break')

    def visit_Continue(self, _):
        return Statement('continue')

    def visit_BoolOp(self, node):
        values = [self.visit(value) for value in node.values]
        op = operator_to_lambda[type(node.op)]
        return reduce(op, values)

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if ispowi(node):
            right = 'std::integral_constant<long, {}>{{}}'.format(node.right.value)
        if isstr(node.left):
            left = 'pythonic::types::str({})'.format(left)
        elif isstr(node.right):
            right = 'pythonic::types::str({})'.format(right)
        return operator_to_lambda[type(node.op)](left, right)

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        return operator_to_lambda[type(node.op)](operand)

    def visit_IfExp(self, node):
        test = self.visit(node.test)
        body = self.visit(node.body)
        orelse = self.visit(node.orelse)
        return '(((bool){0}) ? typename __combined<decltype({1}), decltype({2})>::type({1}) : typename __combined<decltype({1}), decltype({2})>::type({2}))'.format(test, body, orelse)

    def visit_List(self, node):
        if not node.elts:
            return '{}(pythonic::types::empty_list())'.format(self.typeof(node))
        else:
            elts = [self.visit(n) for n in node.elts]
            node_type = self.types[node]
            if len(elts) == 1:
                return '{0}({1}, pythonic::types::single_value())'.format(self.typeof(self.types.builder.Assignable(node_type)), elts[0])
            else:
                return '{0}({{{1}}})'.format(self.typeof(self.types.builder.Assignable(node_type)), ', '.join(elts))

    def visit_Set(self, node):
        if not node.elts:
            return '{}(pythonic::types::empty_set())'.format(self.typeof(node))
        else:
            elts = [self.visit(n) for n in node.elts]
            node_type = self.types.builder.Assignable(self.types[node])
            if len(elts) == 1:
                return '{0}({1}, pythonic::types::single_value())'.format(self.typeof(node_type), elts[0])
            else:
                return '{0}{{{{{1}}}}}'.format(self.typeof(node_type), ', '.join(('static_cast<typename {}::value_type>({})'.format(self.typeof(node_type), elt) for elt in elts)))

    def visit_Dict(self, node):
        if not node.keys:
            return '{}(pythonic::types::empty_dict())'.format(self.typeof(node))
        else:
            keys = [self.visit(n) for n in node.keys]
            values = [self.visit(n) for n in node.values]
            return '{0}{{{{{1}}}}}'.format(self.typeof(self.types.builder.Assignable(self.types[node])), ', '.join(('{{ {0}, {1} }}'.format(k, v) for k, v in zip(keys, values))))

    def visit_Tuple(self, node):
        elts = [self.visit(elt) for elt in node.elts]
        tuple_type = self.types[node]
        result = 'pythonic::types::make_tuple({0})'.format(', '.join(elts))
        if isinstance(tuple_type, self.types.builder.CombinedTypes):
            return '({}){}'.format(self.typeof(tuple_type), result)
        else:
            return result

    def visit_Compare(self, node):
        left = self.visit(node.left)
        ops = [operator_to_lambda[type(n)] for n in node.ops]
        comparators = [self.visit(n) for n in node.comparators]
        all_cmps = zip([left] + comparators[:-1], ops, comparators)
        return ' and '.join((op(x, y) for x, op, y in all_cmps))

    def visit_Call(self, node):
        args = [self.visit(n) for n in node.args]
        func = self.visit(node.func)
        if func == 'pythonic::builtins::functor::getattr{}':
            attrname = node.args[1].value
            fmt = 'pythonic::builtins::getattr({}{{}}, {})'
            attr = 'pythonic::types::attr::' + attrname.upper()
            if attributes_table[attrname][1].isstatic() and node in self.immediates:
                arg = '(decltype(&{}))nullptr'.format(args[0])
            else:
                arg = args[0]
            result = fmt.format(attr, arg)
        elif args and self.is_user_function(node.func):
            result = 'pythonic::types::call({})'.format(', '.join([func] + args))
        else:
            result = '{}({})'.format(func, ', '.join(args))
        if isinstance(self.types.get(node), self.types.builder.CombinedTypes):
            return '({}){}'.format(self.typeof(node), result)
        else:
            return result

    def visit_Constant(self, node):
        if node.value is None:
            ret = 'pythonic::builtins::None'
        elif isinstance(node.value, bool):
            ret = str(node.value).lower()
        elif isinstance(node.value, str):
            quoted = quote_cxxstring(node.value)
            if len(node.value) == 1:
                quoted = quoted.replace("'", "\\'")
                ret = "pythonic::types::chr('" + quoted + "')"
            else:
                ret = 'pythonic::types::str("' + quoted + '")'
        elif isinstance(node.value, complex):
            ret = '{0}({1}, {2})'.format(PYTYPE_TO_CTYPE_TABLE[complex], node.value.real, node.value.imag)
        elif isnan(node.value):
            ret = 'pythonic::numpy::nan'
        elif isinf(node.value):
            ret = ('+' if node.value >= 0 else '-') + 'pythonic::numpy::inf'
        else:
            ret = repr(node.value) + TYPE_TO_SUFFIX.get(type(node.value), '')
        if node in self.immediates:
            assert isinstance(node.value, int)
            return 'std::integral_constant<%s, %s>{}' % (PYTYPE_TO_CTYPE_TABLE[type(node.value)], str(node.value).lower())
        return ret

    def visit_Attribute(self, node):
        obj, path = attr_to_path(node)
        sattr = '::'.join(map(cxxid, path))
        if not obj.isliteral():
            sattr += '{}'
        return sattr

    def all_positive(self, node):
        if isinstance(node, ast.Tuple):
            return all((self.range_values[elt].low >= 0 for elt in node.elts))
        return self.range_values[node].low >= 0

    def stores_to(self, node):
        ancestors = self.ancestors[node] + (node,)
        stmt_indices = [i for i, n in enumerate(ancestors) if isinstance(n, (ast.Assign, ast.For))]
        if not stmt_indices:
            return True
        stmt_index = stmt_indices[-1]
        if isinstance(ancestors[stmt_index], ast.Assign):
            return ancestors[stmt_index + 1] is ancestors[stmt_index].value
        else:
            return ancestors[stmt_index + 1] is not ancestors[stmt_index].target

    def visit_Subscript(self, node):
        value = self.visit(node.value)
        if self.stores_to(node):
            value = 'pythonic::types::as_const({})'.format(value)
        if isstr(node.value):
            value = 'pythonic::types::str({})'.format(value)
        if isnum(node.slice) and node.slice.value >= 0 and isinstance(node.slice.value, int):
            return 'std::get<{0}>({1})'.format(node.slice.value, value)
        elif self.all_positive(node.slice):
            slice_ = self.visit(node.slice)
            return '{1}.fast({0})'.format(slice_, value)
        elif isextslice(node.slice):
            slices = [self.visit(elt) for elt in node.slice.elts]
            return '{1}({0})'.format(','.join(slices), value)
        else:
            slice_ = self.visit(node.slice)
            return '{1}[{0}]'.format(slice_, value)

    def visit_Name(self, node):
        if node.id in self.local_names:
            return cxxid(node.id)
        elif node.id in self.global_declarations:
            return '{0}()'.format(cxxid(node.id))
        else:
            return cxxid(node.id)

    def visit_Slice(self, node):
        args = []
        for field in ('lower', 'upper', 'step'):
            nfield = getattr(node, field)
            arg = self.visit(nfield) if nfield else 'pythonic::builtins::None'
            args.append(arg)
        nstep = node.step
        if nstep is None or (isnum(nstep) and nstep.value > 0):
            if nstep is None or nstep.value == 1:
                if self.all_positive(node.lower) and self.all_positive(node.upper):
                    builder = 'pythonic::types::fast_contiguous_slice({0},{1})'
                else:
                    builder = 'pythonic::types::contiguous_slice({0},{1})'
                step = 1
            else:
                builder = 'pythonic::types::cstride_slice<{2}>({0},{1})'
                step = nstep.value
            return builder.format(args[0], args[1], step)
        else:
            return 'pythonic::types::slice({},{},{})'.format(*args)