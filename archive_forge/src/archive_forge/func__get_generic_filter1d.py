import cupy
from cupy_backends.cuda.api import runtime
from cupy import _util
from cupyx.scipy.ndimage import _filters_core
@_util.memoize(for_each_device=True)
def _get_generic_filter1d(rk, length, n_lines, filter_size, origin, mode, cval, in_ctype, out_ctype, int_type):
    """
    The generic 1d filter is different than all other filters and thus is the
    only filter that doesn't use _generate_nd_kernel() and has a completely
    custom raw kernel.
    """
    in_length = length + filter_size - 1
    start = filter_size // 2 + origin
    end = start + length
    if mode == 'constant':
        boundary, boundary_early = ('', '\n        for (idx_t j = 0; j < {start}; ++j) {{ input_line[j] = {cval}; }}\n        for (idx_t j = {end}; j<{in_length}; ++j) {{ input_line[j] = {cval}; }}\n        '.format(start=start, end=end, in_length=in_length, cval=cval))
    else:
        if length == 1:
            a = b = 'j_ = 0;'
        elif mode == 'reflect':
            j = 'j_ = ({j}) % ({length} * 2);\nj_ = min(j_, 2 * {length} - 1 - j_);'
            a = j.format(j='-1 - j_', length=length)
            b = j.format(j='j_', length=length)
        elif mode == 'mirror':
            j = 'j_ = 1 + (({j}) - 1) % (({length} - 1) * 2);\nj_ = min(j_, 2 * {length} - 2 - j_);'
            a = j.format(j='-j_', length=length)
            b = j.format(j='j_', length=length)
        elif mode == 'nearest':
            a, b = ('j_ = 0;', 'j_ = {length}-1;'.format(length=length))
        elif mode == 'wrap':
            a = 'j_ = j_ % {length} + {length};'.format(length=length)
            b = 'j_ = j_ % {length};'.format(length=length)
        loop = 'for (idx_t j = {{}}; j < {{}}; ++j) {{{{\n            idx_t j_ = j - {start};\n            {{}}\n            input_line[j] = input_line[j_ + {start}];\n        }}}}'.format(start=start)
        boundary_early = ''
        boundary = loop.format(0, start, a) + '\n' + loop.format(end, in_length, b)
    name = 'generic1d_{}_{}_{}'.format(length, filter_size, rk.name)
    code = '#include "cupy/carray.cuh"\n#include "cupy/complex.cuh"\n{include_type_traits}  // let Jitify handle this\n\nnamespace raw_kernel {{\n{rk_code}\n}}\n\n{CAST}\n\ntypedef unsigned char byte;\ntypedef {in_ctype} X;\ntypedef {out_ctype} Y;\ntypedef {int_type} idx_t;\n\n__device__ idx_t offset(idx_t i, idx_t axis, idx_t ndim,\n                        const idx_t* shape, const idx_t* strides) {{\n    idx_t index = 0;\n    for (idx_t a = ndim; --a > 0; ) {{\n        if (a == axis) {{ continue; }}\n        index += (i % shape[a]) * strides[a];\n        i /= shape[a];\n    }}\n    return index + strides[0] * i;\n}}\n\nextern "C" __global__\nvoid {name}(const byte* input, byte* output, const idx_t* x) {{\n    const idx_t axis = x[0], ndim = x[1],\n        *shape = x+2, *in_strides = x+2+ndim, *out_strides = x+2+2*ndim;\n\n    const idx_t in_elem_stride = in_strides[axis];\n    const idx_t out_elem_stride = out_strides[axis];\n\n    double input_line[{in_length}];\n    double output_line[{length}];\n    {boundary_early}\n\n    for (idx_t i = ((idx_t)blockIdx.x) * blockDim.x + threadIdx.x;\n            i < {n_lines};\n            i += ((idx_t)blockDim.x) * gridDim.x) {{\n        // Copy line from input (with boundary filling)\n        const byte* input_ = input + offset(i, axis, ndim, shape, in_strides);\n        for (idx_t j = 0; j < {length}; ++j) {{\n            input_line[j+{start}] = (double)*(X*)(input_+j*in_elem_stride);\n        }}\n        {boundary}\n\n        raw_kernel::{rk_name}(input_line, {in_length}, output_line, {length});\n\n        // Copy line to output\n        byte* output_ = output + offset(i, axis, ndim, shape, out_strides);\n        for (idx_t j = 0; j < {length}; ++j) {{\n            *(Y*)(output_+j*out_elem_stride) = cast<Y>(output_line[j]);\n        }}\n    }}\n}}'.format(n_lines=n_lines, length=length, in_length=in_length, start=start, in_ctype=in_ctype, out_ctype=out_ctype, int_type=int_type, boundary_early=boundary_early, boundary=boundary, name=name, rk_name=rk.name, rk_code=rk.code.replace('__global__', '__device__'), include_type_traits='' if runtime.is_hip else '#include <type_traits>\n', CAST=_filters_core._CAST_FUNCTION)
    return cupy.RawKernel(code, name, ('--std=c++11',) + rk.options, jitify=True)