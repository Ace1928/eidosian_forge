import cupy
from cupy_backends.cuda.api import runtime
from cupy import _util
from cupyx.scipy.ndimage import _filters_core
@_util.memoize(for_each_device=True)
def _get_generic_filter_red(rk, in_dtype, out_dtype, filter_size, mode, wshape, offsets, cval, int_type):
    """Generic filter implementation based on a reduction kernel."""
    in_param, out_param = (rk.in_params[0], rk.out_params[0])
    out_ctype = out_param.ctype
    if out_param.dtype is None:
        out_ctype = cupy._core._scalar.get_typename(in_dtype if out_param.ctype == in_param.ctype else out_dtype)
    setup = '\n    int iv = 0;\n    X values[{size}];\n    CArray<X, 1, true, true> sub_in(values, {{{size}}});\n    {out_ctype} val_out;\n    CArray<{out_ctype}, 1, true, true> sub_out(&val_out, {{1}});\n    '.format(size=filter_size, out_ctype=out_ctype)
    sub_call = 'reduction_kernel::{}(sub_in, sub_out);\n    y = cast<Y>(val_out);'.format(rk.name)
    sub_kernel = _reduction_kernel_code(rk, filter_size, out_dtype, in_dtype)
    return _filters_core._generate_nd_kernel('generic_{}_{}'.format(filter_size, rk.name), setup, 'values[iv++] = {value};', sub_call, mode, wshape, int_type, offsets, cval, preamble=sub_kernel, options=getattr(rk, 'options', ()))