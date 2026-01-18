import os
import tempfile
from sympy.core.symbol import (Symbol, symbols)
from sympy.codegen.ast import (
from sympy.codegen.fnodes import (
from sympy.codegen.futils import render_as_module
from sympy.core.expr import unchanged
from sympy.external import import_module
from sympy.printing.codeprinter import fcode
from sympy.utilities._compilation import has_fortran, compile_run_strings, compile_link_import_strings
from sympy.utilities._compilation.util import may_xfail
from sympy.testing.pytest import skip, XFAIL
@may_xfail
def test_size_assumed_shape():
    if not has_fortran():
        skip('No fortran compiler found.')
    a = Symbol('a', real=True)
    body = [Return((sum_(a ** 2) / size(a)) ** 0.5)]
    arr = array(a, dim=[':'], intent='in')
    fd = FunctionDefinition(real, 'rms', [arr], body)
    render_as_module([fd], 'mod_rms')
    (stdout, stderr), info = compile_run_strings([('rms.f90', render_as_module([fd], 'mod_rms')), ('main.f90', 'program myprog\nuse mod_rms, only: rms\nreal*8, dimension(4), parameter :: x = [4, 2, 2, 2]\nprint *, dsqrt(7d0) - rms(x)\nend program\n')], clean=True)
    assert '0.00000' in stdout
    assert stderr == ''
    assert info['exit_status'] == os.EX_OK