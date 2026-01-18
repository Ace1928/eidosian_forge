import ast
import copy
import functools
import linecache
import sys
from typing import Any, Dict, List
import triton
@functools.lru_cache(None)
def unroll_varargs(kernel, N: int):
    """
    Specializes a triton kernel with variable number of inputs
    to a specific number of inputs `N`.
    NOTE: Because it's quite costly to call `triton.jit`,
    we cache the returned value with `lru_cache`
    """
    global _FILENAME_TO_SRC, _getlines_orig
    k = triton.JITFunction(kernel.fn)
    parsed = ast.parse(k.src)
    nodeVisitor = _VisitorUnrollKernel(N=N)
    parsed = nodeVisitor.visit(parsed)
    parsed = ast.fix_missing_locations(parsed)
    if (sys.version_info.major, sys.version_info.minor) <= (3, 8):
        raise RuntimeError('Error: This functionality requires python 3.9 or above')
    new_src = ast.unparse(parsed)
    fn_filename = f'<unroll_varargs-{kernel.fn.__name__}-{N}>'
    code = compile(new_src, fn_filename, 'exec')
    _locals: Dict[str, Any] = {}
    exec(code, kernel.fn.__globals__, _locals)
    assert len(_locals) == 1, len(_locals)
    fn = next(iter(_locals.values()))
    if not _FILENAME_TO_SRC:
        _getlines_orig = linecache.getlines
        linecache.getlines = _monkey_patched_getlines
    _FILENAME_TO_SRC[fn_filename] = new_src
    jitted_fn = triton.jit(fn)
    jitted_fn.src = new_src
    return jitted_fn