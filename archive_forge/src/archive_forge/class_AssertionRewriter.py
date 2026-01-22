import ast
from collections import defaultdict
import errno
import functools
import importlib.abc
import importlib.machinery
import importlib.util
import io
import itertools
import marshal
import os
from pathlib import Path
from pathlib import PurePath
import struct
import sys
import tokenize
import types
from typing import Callable
from typing import Dict
from typing import IO
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from _pytest._io.saferepr import DEFAULT_REPR_MAX_SIZE
from _pytest._io.saferepr import saferepr
from _pytest._version import version
from _pytest.assertion import util
from _pytest.config import Config
from _pytest.main import Session
from _pytest.pathlib import absolutepath
from _pytest.pathlib import fnmatch_ex
from _pytest.stash import StashKey
from _pytest.assertion.util import format_explanation as _format_explanation  # noqa:F401, isort:skip
class AssertionRewriter(ast.NodeVisitor):
    """Assertion rewriting implementation.

    The main entrypoint is to call .run() with an ast.Module instance,
    this will then find all the assert statements and rewrite them to
    provide intermediate values and a detailed assertion error.  See
    http://pybites.blogspot.be/2011/07/behind-scenes-of-pytests-new-assertion.html
    for an overview of how this works.

    The entry point here is .run() which will iterate over all the
    statements in an ast.Module and for each ast.Assert statement it
    finds call .visit() with it.  Then .visit_Assert() takes over and
    is responsible for creating new ast statements to replace the
    original assert statement: it rewrites the test of an assertion
    to provide intermediate values and replace it with an if statement
    which raises an assertion error with a detailed explanation in
    case the expression is false and calls pytest_assertion_pass hook
    if expression is true.

    For this .visit_Assert() uses the visitor pattern to visit all the
    AST nodes of the ast.Assert.test field, each visit call returning
    an AST node and the corresponding explanation string.  During this
    state is kept in several instance attributes:

    :statements: All the AST statements which will replace the assert
       statement.

    :variables: This is populated by .variable() with each variable
       used by the statements so that they can all be set to None at
       the end of the statements.

    :variable_counter: Counter to create new unique variables needed
       by statements.  Variables are created using .variable() and
       have the form of "@py_assert0".

    :expl_stmts: The AST statements which will be executed to get
       data from the assertion.  This is the code which will construct
       the detailed assertion message that is used in the AssertionError
       or for the pytest_assertion_pass hook.

    :explanation_specifiers: A dict filled by .explanation_param()
       with %-formatting placeholders and their corresponding
       expressions to use in the building of an assertion message.
       This is used by .pop_format_context() to build a message.

    :stack: A stack of the explanation_specifiers dicts maintained by
       .push_format_context() and .pop_format_context() which allows
       to build another %-formatted string while already building one.

    :scope: A tuple containing the current scope used for variables_overwrite.

    :variables_overwrite: A dict filled with references to variables
       that change value within an assert. This happens when a variable is
       reassigned with the walrus operator

    This state, except the variables_overwrite,  is reset on every new assert
    statement visited and used by the other visitors.
    """

    def __init__(self, module_path: Optional[str], config: Optional[Config], source: bytes) -> None:
        super().__init__()
        self.module_path = module_path
        self.config = config
        if config is not None:
            self.enable_assertion_pass_hook = config.getini('enable_assertion_pass_hook')
        else:
            self.enable_assertion_pass_hook = False
        self.source = source
        self.scope: tuple[ast.AST, ...] = ()
        self.variables_overwrite: defaultdict[tuple[ast.AST, ...], Dict[str, str]] = defaultdict(dict)

    def run(self, mod: ast.Module) -> None:
        """Find all assert statements in *mod* and rewrite them."""
        if not mod.body:
            return
        doc = getattr(mod, 'docstring', None)
        expect_docstring = doc is None
        if doc is not None and self.is_rewrite_disabled(doc):
            return
        pos = 0
        item = None
        for item in mod.body:
            if expect_docstring and isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
                doc = item.value.value
                if self.is_rewrite_disabled(doc):
                    return
                expect_docstring = False
            elif isinstance(item, ast.ImportFrom) and item.level == 0 and (item.module == '__future__'):
                pass
            else:
                break
            pos += 1
        if isinstance(item, ast.FunctionDef) and item.decorator_list:
            lineno = item.decorator_list[0].lineno
        else:
            lineno = item.lineno
        if sys.version_info >= (3, 10):
            aliases = [ast.alias('builtins', '@py_builtins', lineno=lineno, col_offset=0), ast.alias('_pytest.assertion.rewrite', '@pytest_ar', lineno=lineno, col_offset=0)]
        else:
            aliases = [ast.alias('builtins', '@py_builtins'), ast.alias('_pytest.assertion.rewrite', '@pytest_ar')]
        imports = [ast.Import([alias], lineno=lineno, col_offset=0) for alias in aliases]
        mod.body[pos:pos] = imports
        self.scope = (mod,)
        nodes: List[Union[ast.AST, Sentinel]] = [mod]
        while nodes:
            node = nodes.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.scope = tuple((*self.scope, node))
                nodes.append(_SCOPE_END_MARKER)
            if node == _SCOPE_END_MARKER:
                self.scope = self.scope[:-1]
                continue
            assert isinstance(node, ast.AST)
            for name, field in ast.iter_fields(node):
                if isinstance(field, list):
                    new: List[ast.AST] = []
                    for i, child in enumerate(field):
                        if isinstance(child, ast.Assert):
                            new.extend(self.visit(child))
                        else:
                            new.append(child)
                            if isinstance(child, ast.AST):
                                nodes.append(child)
                    setattr(node, name, new)
                elif isinstance(field, ast.AST) and (not isinstance(field, ast.expr)):
                    nodes.append(field)

    @staticmethod
    def is_rewrite_disabled(docstring: str) -> bool:
        return 'PYTEST_DONT_REWRITE' in docstring

    def variable(self) -> str:
        """Get a new variable."""
        name = '@py_assert' + str(next(self.variable_counter))
        self.variables.append(name)
        return name

    def assign(self, expr: ast.expr) -> ast.Name:
        """Give *expr* a name."""
        name = self.variable()
        self.statements.append(ast.Assign([ast.Name(name, ast.Store())], expr))
        return ast.Name(name, ast.Load())

    def display(self, expr: ast.expr) -> ast.expr:
        """Call saferepr on the expression."""
        return self.helper('_saferepr', expr)

    def helper(self, name: str, *args: ast.expr) -> ast.expr:
        """Call a helper in this module."""
        py_name = ast.Name('@pytest_ar', ast.Load())
        attr = ast.Attribute(py_name, name, ast.Load())
        return ast.Call(attr, list(args), [])

    def builtin(self, name: str) -> ast.Attribute:
        """Return the builtin called *name*."""
        builtin_name = ast.Name('@py_builtins', ast.Load())
        return ast.Attribute(builtin_name, name, ast.Load())

    def explanation_param(self, expr: ast.expr) -> str:
        """Return a new named %-formatting placeholder for expr.

        This creates a %-formatting placeholder for expr in the
        current formatting context, e.g. ``%(py0)s``.  The placeholder
        and expr are placed in the current format context so that it
        can be used on the next call to .pop_format_context().
        """
        specifier = 'py' + str(next(self.variable_counter))
        self.explanation_specifiers[specifier] = expr
        return '%(' + specifier + ')s'

    def push_format_context(self) -> None:
        """Create a new formatting context.

        The format context is used for when an explanation wants to
        have a variable value formatted in the assertion message.  In
        this case the value required can be added using
        .explanation_param().  Finally .pop_format_context() is used
        to format a string of %-formatted values as added by
        .explanation_param().
        """
        self.explanation_specifiers: Dict[str, ast.expr] = {}
        self.stack.append(self.explanation_specifiers)

    def pop_format_context(self, expl_expr: ast.expr) -> ast.Name:
        """Format the %-formatted string with current format context.

        The expl_expr should be an str ast.expr instance constructed from
        the %-placeholders created by .explanation_param().  This will
        add the required code to format said string to .expl_stmts and
        return the ast.Name instance of the formatted string.
        """
        current = self.stack.pop()
        if self.stack:
            self.explanation_specifiers = self.stack[-1]
        keys = [ast.Constant(key) for key in current.keys()]
        format_dict = ast.Dict(keys, list(current.values()))
        form = ast.BinOp(expl_expr, ast.Mod(), format_dict)
        name = '@py_format' + str(next(self.variable_counter))
        if self.enable_assertion_pass_hook:
            self.format_variables.append(name)
        self.expl_stmts.append(ast.Assign([ast.Name(name, ast.Store())], form))
        return ast.Name(name, ast.Load())

    def generic_visit(self, node: ast.AST) -> Tuple[ast.Name, str]:
        """Handle expressions we don't have custom code for."""
        assert isinstance(node, ast.expr)
        res = self.assign(node)
        return (res, self.explanation_param(self.display(res)))

    def visit_Assert(self, assert_: ast.Assert) -> List[ast.stmt]:
        """Return the AST statements to replace the ast.Assert instance.

        This rewrites the test of an assertion to provide
        intermediate values and replace it with an if statement which
        raises an assertion error with a detailed explanation in case
        the expression is false.
        """
        if isinstance(assert_.test, ast.Tuple) and len(assert_.test.elts) >= 1:
            import warnings
            from _pytest.warning_types import PytestAssertRewriteWarning
            assert self.module_path is not None
            warnings.warn_explicit(PytestAssertRewriteWarning('assertion is always true, perhaps remove parentheses?'), category=None, filename=self.module_path, lineno=assert_.lineno)
        self.statements: List[ast.stmt] = []
        self.variables: List[str] = []
        self.variable_counter = itertools.count()
        if self.enable_assertion_pass_hook:
            self.format_variables: List[str] = []
        self.stack: List[Dict[str, ast.expr]] = []
        self.expl_stmts: List[ast.stmt] = []
        self.push_format_context()
        top_condition, explanation = self.visit(assert_.test)
        negation = ast.UnaryOp(ast.Not(), top_condition)
        if self.enable_assertion_pass_hook:
            msg = self.pop_format_context(ast.Constant(explanation))
            if assert_.msg:
                assertmsg = self.helper('_format_assertmsg', assert_.msg)
                gluestr = '\n>assert '
            else:
                assertmsg = ast.Constant('')
                gluestr = 'assert '
            err_explanation = ast.BinOp(ast.Constant(gluestr), ast.Add(), msg)
            err_msg = ast.BinOp(assertmsg, ast.Add(), err_explanation)
            err_name = ast.Name('AssertionError', ast.Load())
            fmt = self.helper('_format_explanation', err_msg)
            exc = ast.Call(err_name, [fmt], [])
            raise_ = ast.Raise(exc, None)
            statements_fail = []
            statements_fail.extend(self.expl_stmts)
            statements_fail.append(raise_)
            fmt_pass = self.helper('_format_explanation', msg)
            orig = _get_assertion_exprs(self.source)[assert_.lineno]
            hook_call_pass = ast.Expr(self.helper('_call_assertion_pass', ast.Constant(assert_.lineno), ast.Constant(orig), fmt_pass))
            hook_impl_test = ast.If(self.helper('_check_if_assertion_pass_impl'), [*self.expl_stmts, hook_call_pass], [])
            statements_pass = [hook_impl_test]
            main_test = ast.If(negation, statements_fail, statements_pass)
            self.statements.append(main_test)
            if self.format_variables:
                variables = [ast.Name(name, ast.Store()) for name in self.format_variables]
                clear_format = ast.Assign(variables, ast.Constant(None))
                self.statements.append(clear_format)
        else:
            body = self.expl_stmts
            self.statements.append(ast.If(negation, body, []))
            if assert_.msg:
                assertmsg = self.helper('_format_assertmsg', assert_.msg)
                explanation = '\n>assert ' + explanation
            else:
                assertmsg = ast.Constant('')
                explanation = 'assert ' + explanation
            template = ast.BinOp(assertmsg, ast.Add(), ast.Constant(explanation))
            msg = self.pop_format_context(template)
            fmt = self.helper('_format_explanation', msg)
            err_name = ast.Name('AssertionError', ast.Load())
            exc = ast.Call(err_name, [fmt], [])
            raise_ = ast.Raise(exc, None)
            body.append(raise_)
        if self.variables:
            variables = [ast.Name(name, ast.Store()) for name in self.variables]
            clear = ast.Assign(variables, ast.Constant(None))
            self.statements.append(clear)
        for stmt in self.statements:
            for node in traverse_node(stmt):
                ast.copy_location(node, assert_)
        return self.statements

    def visit_NamedExpr(self, name: ast.NamedExpr) -> Tuple[ast.NamedExpr, str]:
        locs = ast.Call(self.builtin('locals'), [], [])
        target_id = name.target.id
        inlocs = ast.Compare(ast.Constant(target_id), [ast.In()], [locs])
        dorepr = self.helper('_should_repr_global_name', name)
        test = ast.BoolOp(ast.Or(), [inlocs, dorepr])
        expr = ast.IfExp(test, self.display(name), ast.Constant(target_id))
        return (name, self.explanation_param(expr))

    def visit_Name(self, name: ast.Name) -> Tuple[ast.Name, str]:
        locs = ast.Call(self.builtin('locals'), [], [])
        inlocs = ast.Compare(ast.Constant(name.id), [ast.In()], [locs])
        dorepr = self.helper('_should_repr_global_name', name)
        test = ast.BoolOp(ast.Or(), [inlocs, dorepr])
        expr = ast.IfExp(test, self.display(name), ast.Constant(name.id))
        return (name, self.explanation_param(expr))

    def visit_BoolOp(self, boolop: ast.BoolOp) -> Tuple[ast.Name, str]:
        res_var = self.variable()
        expl_list = self.assign(ast.List([], ast.Load()))
        app = ast.Attribute(expl_list, 'append', ast.Load())
        is_or = int(isinstance(boolop.op, ast.Or))
        body = save = self.statements
        fail_save = self.expl_stmts
        levels = len(boolop.values) - 1
        self.push_format_context()
        for i, v in enumerate(boolop.values):
            if i:
                fail_inner: List[ast.stmt] = []
                self.expl_stmts.append(ast.If(cond, fail_inner, []))
                self.expl_stmts = fail_inner
                if isinstance(v, ast.Compare) and isinstance(v.left, ast.NamedExpr) and (v.left.target.id in [ast_expr.id for ast_expr in boolop.values[:i] if hasattr(ast_expr, 'id')]):
                    pytest_temp = self.variable()
                    self.variables_overwrite[self.scope][v.left.target.id] = v.left
                    v.left.target.id = pytest_temp
            self.push_format_context()
            res, expl = self.visit(v)
            body.append(ast.Assign([ast.Name(res_var, ast.Store())], res))
            expl_format = self.pop_format_context(ast.Constant(expl))
            call = ast.Call(app, [expl_format], [])
            self.expl_stmts.append(ast.Expr(call))
            if i < levels:
                cond: ast.expr = res
                if is_or:
                    cond = ast.UnaryOp(ast.Not(), cond)
                inner: List[ast.stmt] = []
                self.statements.append(ast.If(cond, inner, []))
                self.statements = body = inner
        self.statements = save
        self.expl_stmts = fail_save
        expl_template = self.helper('_format_boolop', expl_list, ast.Constant(is_or))
        expl = self.pop_format_context(expl_template)
        return (ast.Name(res_var, ast.Load()), self.explanation_param(expl))

    def visit_UnaryOp(self, unary: ast.UnaryOp) -> Tuple[ast.Name, str]:
        pattern = UNARY_MAP[unary.op.__class__]
        operand_res, operand_expl = self.visit(unary.operand)
        res = self.assign(ast.UnaryOp(unary.op, operand_res))
        return (res, pattern % (operand_expl,))

    def visit_BinOp(self, binop: ast.BinOp) -> Tuple[ast.Name, str]:
        symbol = BINOP_MAP[binop.op.__class__]
        left_expr, left_expl = self.visit(binop.left)
        right_expr, right_expl = self.visit(binop.right)
        explanation = f'({left_expl} {symbol} {right_expl})'
        res = self.assign(ast.BinOp(left_expr, binop.op, right_expr))
        return (res, explanation)

    def visit_Call(self, call: ast.Call) -> Tuple[ast.Name, str]:
        new_func, func_expl = self.visit(call.func)
        arg_expls = []
        new_args = []
        new_kwargs = []
        for arg in call.args:
            if isinstance(arg, ast.Name) and arg.id in self.variables_overwrite.get(self.scope, {}):
                arg = self.variables_overwrite[self.scope][arg.id]
            res, expl = self.visit(arg)
            arg_expls.append(expl)
            new_args.append(res)
        for keyword in call.keywords:
            if isinstance(keyword.value, ast.Name) and keyword.value.id in self.variables_overwrite.get(self.scope, {}):
                keyword.value = self.variables_overwrite[self.scope][keyword.value.id]
            res, expl = self.visit(keyword.value)
            new_kwargs.append(ast.keyword(keyword.arg, res))
            if keyword.arg:
                arg_expls.append(keyword.arg + '=' + expl)
            else:
                arg_expls.append('**' + expl)
        expl = '{}({})'.format(func_expl, ', '.join(arg_expls))
        new_call = ast.Call(new_func, new_args, new_kwargs)
        res = self.assign(new_call)
        res_expl = self.explanation_param(self.display(res))
        outer_expl = f'{res_expl}\n{{{res_expl} = {expl}\n}}'
        return (res, outer_expl)

    def visit_Starred(self, starred: ast.Starred) -> Tuple[ast.Starred, str]:
        res, expl = self.visit(starred.value)
        new_starred = ast.Starred(res, starred.ctx)
        return (new_starred, '*' + expl)

    def visit_Attribute(self, attr: ast.Attribute) -> Tuple[ast.Name, str]:
        if not isinstance(attr.ctx, ast.Load):
            return self.generic_visit(attr)
        value, value_expl = self.visit(attr.value)
        res = self.assign(ast.Attribute(value, attr.attr, ast.Load()))
        res_expl = self.explanation_param(self.display(res))
        pat = '%s\n{%s = %s.%s\n}'
        expl = pat % (res_expl, res_expl, value_expl, attr.attr)
        return (res, expl)

    def visit_Compare(self, comp: ast.Compare) -> Tuple[ast.expr, str]:
        self.push_format_context()
        if isinstance(comp.left, ast.Name) and comp.left.id in self.variables_overwrite.get(self.scope, {}):
            comp.left = self.variables_overwrite[self.scope][comp.left.id]
        if isinstance(comp.left, ast.NamedExpr):
            self.variables_overwrite[self.scope][comp.left.target.id] = comp.left
        left_res, left_expl = self.visit(comp.left)
        if isinstance(comp.left, (ast.Compare, ast.BoolOp)):
            left_expl = f'({left_expl})'
        res_variables = [self.variable() for i in range(len(comp.ops))]
        load_names = [ast.Name(v, ast.Load()) for v in res_variables]
        store_names = [ast.Name(v, ast.Store()) for v in res_variables]
        it = zip(range(len(comp.ops)), comp.ops, comp.comparators)
        expls = []
        syms = []
        results = [left_res]
        for i, op, next_operand in it:
            if isinstance(next_operand, ast.NamedExpr) and isinstance(left_res, ast.Name) and (next_operand.target.id == left_res.id):
                next_operand.target.id = self.variable()
                self.variables_overwrite[self.scope][left_res.id] = next_operand
            next_res, next_expl = self.visit(next_operand)
            if isinstance(next_operand, (ast.Compare, ast.BoolOp)):
                next_expl = f'({next_expl})'
            results.append(next_res)
            sym = BINOP_MAP[op.__class__]
            syms.append(ast.Constant(sym))
            expl = f'{left_expl} {sym} {next_expl}'
            expls.append(ast.Constant(expl))
            res_expr = ast.Compare(left_res, [op], [next_res])
            self.statements.append(ast.Assign([store_names[i]], res_expr))
            left_res, left_expl = (next_res, next_expl)
        expl_call = self.helper('_call_reprcompare', ast.Tuple(syms, ast.Load()), ast.Tuple(load_names, ast.Load()), ast.Tuple(expls, ast.Load()), ast.Tuple(results, ast.Load()))
        if len(comp.ops) > 1:
            res: ast.expr = ast.BoolOp(ast.And(), load_names)
        else:
            res = load_names[0]
        return (res, self.explanation_param(self.pop_format_context(expl_call)))