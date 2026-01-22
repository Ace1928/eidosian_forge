from os import walk, sep, pardir
from os.path import split, join, abspath, exists, isfile
from glob import glob
import re
import random
import ast
from sympy.testing.pytest import raises
from sympy.testing.quality_unicode import _test_this_file_encoding
class _Visit(ast.NodeVisitor):
    """return the line number corresponding to the
    line on which a bare expression appears if it is a binary op
    or a comparison that is not in a with block.

    EXAMPLES
    ========

    >>> import ast
    >>> class _Visit(ast.NodeVisitor):
    ...     def visit_Expr(self, node):
    ...         if isinstance(node.value, (ast.BinOp, ast.Compare)):
    ...             print(node.lineno)
    ...     def visit_With(self, node):
    ...         pass  # no checking there
    ...
    >>> code='''x = 1    # line 1
    ... for i in range(3):
    ...     x == 2       # <-- 3
    ... if x == 2:
    ...     x == 3       # <-- 5
    ...     x + 1        # <-- 6
    ...     x = 1
    ...     if x == 1:
    ...         print(1)
    ... while x != 1:
    ...     x == 1       # <-- 11
    ... with raises(TypeError):
    ...     c == 1
    ...     raise TypeError
    ... assert x == 1
    ... '''
    >>> _Visit().visit(ast.parse(code))
    3
    5
    6
    11
    """

    def visit_Expr(self, node):
        if isinstance(node.value, (ast.BinOp, ast.Compare)):
            assert None, message_bare_expr % ('', node.lineno)

    def visit_With(self, node):
        pass