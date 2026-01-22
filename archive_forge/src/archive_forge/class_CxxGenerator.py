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
class CxxGenerator(CxxFunction):
    StateHolder = '__generator_state'
    StateValue = '__generator_value'
    FinalStatement = 'that_is_all_folks'

    def process_locals(self, node, node_visited, *skipped):
        return node_visited

    def prepare_functiondef_context(self, node):
        self.extra_declarations = []
        self.yields = {k: (1 + v, 'yield_point{0}'.format(1 + v)) for v, k in enumerate(self.gather(YieldPoints, node))}
        return super(CxxGenerator, self).prepare_functiondef_context(node)

    def visit_FunctionDef(self, node):
        self.returns = False
        tmp = self.prepare_functiondef_context(node)
        operator_body, formal_types, formal_args = tmp
        tmp = self.prepare_types(node)
        dflt_argv, dflt_argt, result_type, callable_type, pure_type = tmp
        next_name = '__generator__{0}'.format(cxxid(node.name))
        instanciated_next_name = '{0}{1}'.format(next_name, '<{0}>'.format(', '.join(formal_types)) if formal_types else '')
        if self.returns:
            operator_body.append(Label(CxxGenerator.FinalStatement))
        operator_body.append(Statement('return result_type()'))
        next_declaration = [FunctionDeclaration(Value('result_type', 'next'), []), EmptyStatement()]
        next_constructors = [FunctionBody(FunctionDeclaration(Value('', next_name), []), Line(': pythonic::yielder() {}'))]
        if formal_types:
            if dflt_argv and all(dflt_argv):
                next_constructors = list()
            next_constructors.append(FunctionBody(make_function_declaration(self, node, '', next_name, formal_types, formal_args, dflt_argv), Line(': {0} {{ }}'.format(', '.join(['pythonic::yielder()'] + ['{0}({0})'.format(arg) for arg in formal_args])))))
        next_iterator = [FunctionBody(FunctionDeclaration(Value('void', 'operator++'), []), Block([Statement('next()')])), FunctionBody(FunctionDeclaration(Value('typename {0}::result_type'.format(instanciated_next_name), 'operator*'), [], 'const'), Block([ReturnStatement(CxxGenerator.StateValue)])), FunctionBody(FunctionDeclaration(Value('pythonic::types::generator_iterator<{0}>'.format(next_name), 'begin'), []), Block([Statement('next()'), ReturnStatement('pythonic::types::generator_iterator<{0}>(*this)'.format(next_name))])), FunctionBody(FunctionDeclaration(Value('pythonic::types::generator_iterator<{0}>'.format(next_name), 'end'), []), Block([ReturnStatement('pythonic::types::generator_iterator<{0}>()'.format(next_name))]))]
        next_signature = templatize(FunctionDeclaration(Value('typename {0}::result_type'.format(instanciated_next_name), '{0}::next'.format(instanciated_next_name)), []), formal_types)
        next_body = operator_body
        next_body.insert(0, Statement('switch({0}) {{ {1} }}'.format(CxxGenerator.StateHolder, ' '.join(('case {0}: goto {1};'.format(num, where) for num, where in sorted(self.yields.values(), key=lambda x: x[0]))))))
        ctx = CachedTypeVisitor(self.lctx)
        next_members = [Statement('{0} {1}'.format(ft, fa)) for ft, fa in zip(formal_types, formal_args)] + [Statement('{0} {1}'.format(ctx(self.types[self.local_names[k]]), k)) for k in self.ldecls] + [Statement('{0} {1}'.format(v, k)) for k, v in self.extra_declarations] + [Statement('typename {0}::result_type {1}'.format(instanciated_next_name, CxxGenerator.StateValue))]
        extern_typedefs = [Typedef(Value(ctx(t), t.name)) for t in self.types[node][1]]
        iterator_typedef = [Typedef(Value('pythonic::types::generator_iterator<{0}>'.format('{0}<{1}>'.format(next_name, ', '.join(formal_types)) if formal_types else next_name), 'iterator')), Typedef(Value(ctx(result_type), 'value_type'))]
        result_typedef = [Typedef(Value(ctx(result_type), 'result_type'))]
        extra_typedefs = ctx.typedefs() + extern_typedefs + iterator_typedef + result_typedef
        next_struct = templatize(Struct(next_name, extra_typedefs + next_members + next_constructors + next_iterator + next_declaration, 'pythonic::yielder'), formal_types)
        next_definition = FunctionBody(next_signature, Block(next_body))
        operator_declaration = [templatize(make_const_function_declaration(self, node, instanciated_next_name, 'operator()', formal_types, formal_args, dflt_argv), formal_types, dflt_argt), EmptyStatement()]
        operator_signature = make_const_function_declaration(self, node, instanciated_next_name, '{0}::operator()'.format(cxxid(node.name)), formal_types, formal_args)
        operator_definition = FunctionBody(templatize(operator_signature, formal_types), Block([ReturnStatement('{0}({1})'.format(instanciated_next_name, ', '.join(formal_args)))]))
        topstruct_type = templatize(Struct('type', extra_typedefs), formal_types)
        topstruct = Struct(cxxid(node.name), [topstruct_type, callable_type, pure_type] + operator_declaration)
        return ([next_struct, topstruct], [next_definition, operator_definition])

    def visit_Return(self, node):
        self.returns = True
        return Block([Statement('{0} = -1'.format(CxxGenerator.StateHolder)), Statement('goto {0}'.format(CxxGenerator.FinalStatement))])

    def visit_Yield(self, node):
        num, label = self.yields[node]
        return ''.join((n for n in Block([Assign(CxxGenerator.StateHolder, num), ReturnStatement('{0} = {1}'.format(CxxGenerator.StateValue, self.visit(node.value))), Statement('{0}:'.format(label))]).generate()))

    def visit_Assign(self, node):
        value = self.visit(node.value)
        targets = [self.visit(t) for t in node.targets]
        alltargets = '= '.join(targets)
        stmt = Assign(alltargets, value)
        return self.process_omp_attachements(node, stmt)

    def can_use_autofor(self, node):
        """
        TODO : Yield should block only if it is use in the for loop, not in the
               whole function.
        """
        return False

    def make_assign(self, local_iter_decl, local_iter, iterable):
        self.extra_declarations.append((local_iter, local_iter_decl))
        return super(CxxGenerator, self).make_assign('', local_iter, iterable)