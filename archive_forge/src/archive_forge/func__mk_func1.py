import tempfile
from sympy.external import import_module
from sympy.printing.codeprinter import ccode
from sympy.utilities._compilation import compile_link_import_strings, has_c
from sympy.utilities._compilation.util import may_xfail
from sympy.testing.pytest import skip
from sympy.codegen.ast import (
from sympy.codegen.cnodes import void, PreIncrement
from sympy.codegen.cutils import render_as_source_file
def _mk_func1():
    declars = n, inp, out = (Variable('n', integer), Pointer('inp', real), Pointer('out', real))
    i = Variable('i', integer)
    whl = While(i < n, [Assignment(out[i], inp[i]), PreIncrement(i)])
    body = CodeBlock(i.as_Declaration(value=0), whl)
    return FunctionDefinition(void, 'our_test_function', declars, body)