from genshi.compat import ast as _ast, _ast_Constant, IS_PYTHON2, isstring, \
class ASTCodeGenerator(object):
    """General purpose base class for AST transformations.

    Every visitor method can be overridden to return an AST node that has been
    altered or replaced in some way.
    """

    def __init__(self, tree):
        self.lines_info = []
        self.line_info = None
        self.code = ''
        self.line = None
        self.last = None
        self.indent = 0
        self.blame_stack = []
        self.visit(tree)
        if self.line.strip():
            self.code += self.line + '\n'
            self.lines_info.append(self.line_info)
        self.line = None
        self.line_info = None

    def _change_indent(self, delta):
        self.indent += delta

    def _new_line(self):
        if self.line is not None:
            self.code += self.line + '\n'
            self.lines_info.append(self.line_info)
        self.line = ' ' * 4 * self.indent
        if len(self.blame_stack) == 0:
            self.line_info = []
            self.last = None
        else:
            self.line_info = [(0, self.blame_stack[-1])]
            self.last = self.blame_stack[-1]

    def _write(self, s):
        if len(s) == 0:
            return
        if len(self.blame_stack) == 0:
            if self.last is not None:
                self.last = None
                self.line_info.append((len(self.line), self.last))
        elif self.last != self.blame_stack[-1]:
            self.last = self.blame_stack[-1]
            self.line_info.append((len(self.line), self.last))
        self.line += s

    def visit(self, node):
        if node is None:
            return None
        if type(node) is tuple:
            return tuple([self.visit(n) for n in node])
        try:
            self.blame_stack.append((node.lineno, node.col_offset))
            info = True
        except AttributeError:
            info = False
        if isinstance(node, (bool, bytes, float, int, str)):
            node = _ast_Constant(node)
        visitor = getattr(self, 'visit_%s' % node.__class__.__name__, None)
        if visitor is None:
            raise Exception('Unhandled node type %r' % type(node))
        ret = visitor(node)
        if info:
            self.blame_stack.pop()
        return ret

    def visit_Module(self, node):
        for n in node.body:
            self.visit(n)
    visit_Interactive = visit_Module
    visit_Suite = visit_Module

    def visit_Expression(self, node):
        self._new_line()
        return self.visit(node.body)

    def visit_arguments(self, node):

        def write_possible_comma():
            if _first[0]:
                _first[0] = False
            else:
                self._write(', ')
        _first = [True]

        def write_args(args, defaults):
            no_default_count = len(args) - len(defaults)
            for i, arg in enumerate(args):
                write_possible_comma()
                self.visit(arg)
                default_idx = i - no_default_count
                if default_idx >= 0 and defaults[default_idx] is not None:
                    self._write('=')
                    self.visit(defaults[i - no_default_count])
        write_args(node.args, node.defaults)
        if getattr(node, 'vararg', None):
            write_possible_comma()
            self._write('*')
            if isstring(node.vararg):
                self._write(node.vararg)
            else:
                self.visit(node.vararg)
        if getattr(node, 'kwonlyargs', None):
            write_args(node.kwonlyargs, node.kw_defaults)
        if getattr(node, 'kwarg', None):
            write_possible_comma()
            self._write('**')
            if isstring(node.kwarg):
                self._write(node.kwarg)
            else:
                self.visit(node.kwarg)
    if not IS_PYTHON2:

        def visit_arg(self, node):
            self._write(node.arg)

    def visit_Starred(self, node):
        self._write('*')
        self.visit(node.value)

    def visit_FunctionDef(self, node):
        decarators = ()
        if hasattr(node, 'decorator_list'):
            decorators = getattr(node, 'decorator_list')
        else:
            decorators = getattr(node, 'decorators', ())
        for decorator in decorators:
            self._new_line()
            self._write('@')
            self.visit(decorator)
        self._new_line()
        self._write('def ' + node.name + '(')
        self.visit(node.args)
        self._write('):')
        self._change_indent(1)
        for statement in node.body:
            self.visit(statement)
        self._change_indent(-1)

    def visit_ClassDef(self, node):
        self._new_line()
        self._write('class ' + node.name)
        if node.bases:
            self._write('(')
            self.visit(node.bases[0])
            for base in node.bases[1:]:
                self._write(', ')
                self.visit(base)
            self._write(')')
        self._write(':')
        self._change_indent(1)
        for statement in node.body:
            self.visit(statement)
        self._change_indent(-1)

    def visit_Return(self, node):
        self._new_line()
        self._write('return')
        if getattr(node, 'value', None):
            self._write(' ')
            self.visit(node.value)

    def visit_Delete(self, node):
        self._new_line()
        self._write('del ')
        self.visit(node.targets[0])
        for target in node.targets[1:]:
            self._write(', ')
            self.visit(target)

    def visit_Assign(self, node):
        self._new_line()
        for target in node.targets:
            self.visit(target)
            self._write(' = ')
        self.visit(node.value)

    def visit_AugAssign(self, node):
        self._new_line()
        self.visit(node.target)
        self._write(' ' + self.binary_operators[node.op.__class__] + '= ')
        self.visit(node.value)

    def visit_Print(self, node):
        self._new_line()
        self._write('print')
        if getattr(node, 'dest', None):
            self._write(' >> ')
            self.visit(node.dest)
            if getattr(node, 'values', None):
                self._write(', ')
        else:
            self._write(' ')
        if getattr(node, 'values', None):
            self.visit(node.values[0])
            for value in node.values[1:]:
                self._write(', ')
                self.visit(value)
        if not node.nl:
            self._write(',')

    def visit_For(self, node):
        self._new_line()
        self._write('for ')
        self.visit(node.target)
        self._write(' in ')
        self.visit(node.iter)
        self._write(':')
        self._change_indent(1)
        for statement in node.body:
            self.visit(statement)
        self._change_indent(-1)
        if getattr(node, 'orelse', None):
            self._new_line()
            self._write('else:')
            self._change_indent(1)
            for statement in node.orelse:
                self.visit(statement)
            self._change_indent(-1)

    def visit_While(self, node):
        self._new_line()
        self._write('while ')
        self.visit(node.test)
        self._write(':')
        self._change_indent(1)
        for statement in node.body:
            self.visit(statement)
        self._change_indent(-1)
        if getattr(node, 'orelse', None):
            self._new_line()
            self._write('else:')
            self._change_indent(1)
            for statement in node.orelse:
                self.visit(statement)
            self._change_indent(-1)

    def visit_If(self, node):
        self._new_line()
        self._write('if ')
        self.visit(node.test)
        self._write(':')
        self._change_indent(1)
        for statement in node.body:
            self.visit(statement)
        self._change_indent(-1)
        if getattr(node, 'orelse', None):
            self._new_line()
            self._write('else:')
            self._change_indent(1)
            for statement in node.orelse:
                self.visit(statement)
            self._change_indent(-1)

    def visit_With(self, node):
        self._new_line()
        self._write('with ')
        items = getattr(node, 'items', None)
        first = True
        if items is None:
            items = [node]
        for item in items:
            if not first:
                self._write(', ')
            first = False
            self.visit(item.context_expr)
            if getattr(item, 'optional_vars', None):
                self._write(' as ')
                self.visit(item.optional_vars)
        self._write(':')
        self._change_indent(1)
        for statement in node.body:
            self.visit(statement)
        self._change_indent(-1)

    def visit_Raise(self, node):
        self._new_line()
        self._write('raise')
        if IS_PYTHON2:
            if not node.type:
                return
            self._write(' ')
            self.visit(node.type)
            if not node.inst:
                return
            self._write(', ')
            self.visit(node.inst)
            if not node.tback:
                return
            self._write(', ')
            self.visit(node.tback)
        else:
            if not node.exc:
                return
            self._write(' ')
            self.visit(node.exc)
            if not node.cause:
                return
            self._write(' from ')
            self.visit(node.cause)

    def visit_TryExcept(self, node):
        self._new_line()
        self._write('try:')
        self._change_indent(1)
        for statement in node.body:
            self.visit(statement)
        self._change_indent(-1)
        if getattr(node, 'handlers', None):
            for handler in node.handlers:
                self.visit(handler)
        self._new_line()
        if getattr(node, 'orelse', None):
            self._write('else:')
            self._change_indent(1)
            for statement in node.orelse:
                self.visit(statement)
            self._change_indent(-1)

    def visit_ExceptHandler(self, node):
        self._new_line()
        self._write('except')
        if getattr(node, 'type', None):
            self._write(' ')
            self.visit(node.type)
        if getattr(node, 'name', None):
            self._write(', ')
            self.visit(node.name)
        self._write(':')
        self._change_indent(1)
        for statement in node.body:
            self.visit(statement)
        self._change_indent(-1)
    visit_excepthandler = visit_ExceptHandler

    def visit_TryFinally(self, node):
        self._new_line()
        self._write('try:')
        self._change_indent(1)
        for statement in node.body:
            self.visit(statement)
        self._change_indent(-1)
        if getattr(node, 'finalbody', None):
            self._new_line()
            self._write('finally:')
            self._change_indent(1)
            for statement in node.finalbody:
                self.visit(statement)
            self._change_indent(-1)

    def visit_Try(self, node):
        self._new_line()
        self._write('try:')
        self._change_indent(1)
        for statement in node.body:
            self.visit(statement)
        self._change_indent(-1)
        if getattr(node, 'handlers', None):
            for handler in node.handlers:
                self.visit(handler)
        self._new_line()
        if getattr(node, 'orelse', None):
            self._write('else:')
            self._change_indent(1)
            for statement in node.orelse:
                self.visit(statement)
            self._change_indent(-1)
        if getattr(node, 'finalbody', None):
            self._new_line()
            self._write('finally:')
            self._change_indent(1)
            for statement in node.finalbody:
                self.visit(statement)
            self._change_indent(-1)

    def visit_Assert(self, node):
        self._new_line()
        self._write('assert ')
        self.visit(node.test)
        if getattr(node, 'msg', None):
            self._write(', ')
            self.visit(node.msg)

    def visit_alias(self, node):
        self._write(node.name)
        if getattr(node, 'asname', None):
            self._write(' as ')
            self._write(node.asname)

    def visit_Import(self, node):
        self._new_line()
        self._write('import ')
        self.visit(node.names[0])
        for name in node.names[1:]:
            self._write(', ')
            self.visit(name)

    def visit_ImportFrom(self, node):
        self._new_line()
        self._write('from ')
        if node.level:
            self._write('.' * node.level)
        self._write(node.module)
        self._write(' import ')
        self.visit(node.names[0])
        for name in node.names[1:]:
            self._write(', ')
            self.visit(name)

    def visit_Exec(self, node):
        self._new_line()
        self._write('exec ')
        self.visit(node.body)
        if not node.globals:
            return
        self._write(', ')
        self.visit(node.globals)
        if not node.locals:
            return
        self._write(', ')
        self.visit(node.locals)

    def visit_Global(self, node):
        self._new_line()
        self._write('global ')
        self.visit(node.names[0])
        for name in node.names[1:]:
            self._write(', ')
            self.visit(name)

    def visit_Expr(self, node):
        self._new_line()
        self.visit(node.value)

    def visit_Pass(self, node):
        self._new_line()
        self._write('pass')

    def visit_Break(self, node):
        self._new_line()
        self._write('break')

    def visit_Continue(self, node):
        self._new_line()
        self._write('continue')

    def with_parens(f):

        def _f(self, node):
            self._write('(')
            f(self, node)
            self._write(')')
        return _f
    bool_operators = {_ast.And: 'and', _ast.Or: 'or'}

    @with_parens
    def visit_BoolOp(self, node):
        joiner = ' ' + self.bool_operators[node.op.__class__] + ' '
        self.visit(node.values[0])
        for value in node.values[1:]:
            self._write(joiner)
            self.visit(value)
    binary_operators = {_ast.Add: '+', _ast.Sub: '-', _ast.Mult: '*', _ast.Div: '/', _ast.Mod: '%', _ast.Pow: '**', _ast.LShift: '<<', _ast.RShift: '>>', _ast.BitOr: '|', _ast.BitXor: '^', _ast.BitAnd: '&', _ast.FloorDiv: '//'}

    @with_parens
    def visit_BinOp(self, node):
        self.visit(node.left)
        self._write(' ' + self.binary_operators[node.op.__class__] + ' ')
        self.visit(node.right)
    unary_operators = {_ast.Invert: '~', _ast.Not: 'not', _ast.UAdd: '+', _ast.USub: '-'}

    def visit_UnaryOp(self, node):
        self._write(self.unary_operators[node.op.__class__] + ' ')
        self.visit(node.operand)

    @with_parens
    def visit_Lambda(self, node):
        self._write('lambda ')
        self.visit(node.args)
        self._write(': ')
        self.visit(node.body)

    @with_parens
    def visit_IfExp(self, node):
        self.visit(node.body)
        self._write(' if ')
        self.visit(node.test)
        self._write(' else ')
        self.visit(node.orelse)

    def visit_Dict(self, node):
        self._write('{')
        for key, value in zip(node.keys, node.values):
            self.visit(key)
            self._write(': ')
            self.visit(value)
            self._write(', ')
        self._write('}')

    def visit_ListComp(self, node):
        self._write('[')
        self.visit(node.elt)
        for generator in node.generators:
            self._write(' for ')
            self.visit(generator.target)
            self._write(' in ')
            self.visit(generator.iter)
            for ifexpr in generator.ifs:
                self._write(' if ')
                self.visit(ifexpr)
        self._write(']')

    def visit_GeneratorExp(self, node):
        self._write('(')
        self.visit(node.elt)
        for generator in node.generators:
            self._write(' for ')
            self.visit(generator.target)
            self._write(' in ')
            self.visit(generator.iter)
            for ifexpr in generator.ifs:
                self._write(' if ')
                self.visit(ifexpr)
        self._write(')')

    def visit_Yield(self, node):
        self._write('yield')
        if getattr(node, 'value', None):
            self._write(' ')
            self.visit(node.value)
    comparision_operators = {_ast.Eq: '==', _ast.NotEq: '!=', _ast.Lt: '<', _ast.LtE: '<=', _ast.Gt: '>', _ast.GtE: '>=', _ast.Is: 'is', _ast.IsNot: 'is not', _ast.In: 'in', _ast.NotIn: 'not in'}

    @with_parens
    def visit_Compare(self, node):
        self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            self._write(' ' + self.comparision_operators[op.__class__] + ' ')
            self.visit(comparator)

    def visit_Call(self, node):
        self.visit(node.func)
        self._write('(')
        first = True
        for arg in node.args:
            if not first:
                self._write(', ')
            first = False
            self.visit(arg)
        for keyword in node.keywords:
            if not first:
                self._write(', ')
            first = False
            if not keyword.arg:
                self._write('**')
            else:
                self._write(keyword.arg)
                self._write('=')
            self.visit(keyword.value)
        if getattr(node, 'starargs', None):
            if not first:
                self._write(', ')
            first = False
            self._write('*')
            self.visit(node.starargs)
        if getattr(node, 'kwargs', None):
            if not first:
                self._write(', ')
            first = False
            self._write('**')
            self.visit(node.kwargs)
        self._write(')')

    def visit_Repr(self, node):
        self._write('`')
        self.visit(node.value)
        self._write('`')

    def visit_Num(self, node):
        self._write(repr(node.n))

    def visit_Str(self, node):
        self._write(repr(node.s))

    def visit_Constant(self, node):
        self._write(repr(node.value))
    if not IS_PYTHON2:

        def visit_Bytes(self, node):
            self._write(repr(node.s))

    def visit_Attribute(self, node):
        self.visit(node.value)
        self._write('.')
        self._write(node.attr)

    def visit_Subscript(self, node):
        self.visit(node.value)
        self._write('[')

        def _process_slice(node):
            if isinstance(node, _ast_Ellipsis):
                self._write('...')
            elif isinstance(node, _ast.Slice):
                if getattr(node, 'lower', 'None'):
                    self.visit(node.lower)
                self._write(':')
                if getattr(node, 'upper', None):
                    self.visit(node.upper)
                if getattr(node, 'step', None):
                    self._write(':')
                    self.visit(node.step)
            elif isinstance(node, _ast.Index):
                self.visit(node.value)
            elif isinstance(node, _ast.ExtSlice):
                self.visit(node.dims[0])
                for dim in node.dims[1:]:
                    self._write(', ')
                    self.visit(dim)
            else:
                self.visit(node)
        _process_slice(node.slice)
        self._write(']')

    def visit_Name(self, node):
        self._write(node.id)

    def visit_NameConstant(self, node):
        if node.value is None:
            self._write('None')
        elif node.value is True:
            self._write('True')
        elif node.value is False:
            self._write('False')
        else:
            raise Exception('Unknown NameConstant %r' % (node.value,))

    def visit_List(self, node):
        self._write('[')
        for elt in node.elts:
            self.visit(elt)
            self._write(', ')
        self._write(']')

    def visit_Tuple(self, node):
        self._write('(')
        for elt in node.elts:
            self.visit(elt) if elt is not None else self._write('None')
            self._write(', ')
        self._write(')')