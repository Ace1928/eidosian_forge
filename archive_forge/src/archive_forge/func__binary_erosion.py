import operator
import warnings
import numpy
import cupy
from cupy import _core
from cupyx.scipy.ndimage import _filters_core
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters
def _binary_erosion(input, structure, iterations, mask, output, border_value, origin, invert, brute_force=True):
    try:
        iterations = operator.index(iterations)
    except TypeError:
        raise TypeError('iterations parameter should be an integer')
    if input.dtype.kind == 'c':
        raise TypeError('Complex type not supported')
    if structure is None:
        structure = generate_binary_structure(input.ndim, 1)
        all_weights_nonzero = input.ndim == 1
        center_is_true = True
        default_structure = True
    else:
        structure = structure.astype(dtype=bool, copy=False)
        default_structure = False
    if structure.ndim != input.ndim:
        raise RuntimeError('structure and input must have same dimensionality')
    if not structure.flags.c_contiguous:
        structure = cupy.ascontiguousarray(structure)
    if structure.size < 1:
        raise RuntimeError('structure must not be empty')
    if mask is not None:
        if mask.shape != input.shape:
            raise RuntimeError('mask and input must have equal sizes')
        if not mask.flags.c_contiguous:
            mask = cupy.ascontiguousarray(mask)
        masked = True
    else:
        masked = False
    origin = _util._fix_sequence_arg(origin, input.ndim, 'origin', int)
    if isinstance(output, cupy.ndarray):
        if output.dtype.kind == 'c':
            raise TypeError('Complex output type not supported')
    else:
        output = bool
    output = _util._get_output(output, input)
    temp_needed = cupy.shares_memory(output, input, 'MAY_SHARE_BOUNDS')
    if temp_needed:
        temp = output
        output = _util._get_output(output.dtype, input)
    if structure.ndim == 0:
        if float(structure):
            output[...] = cupy.asarray(input, dtype=bool)
        else:
            output[...] = ~cupy.asarray(input, dtype=bool)
        return output
    origin = tuple(origin)
    int_type = _util._get_inttype(input)
    offsets = _filters_core._origins_to_offsets(origin, structure.shape)
    if not default_structure:
        nnz = int(cupy.count_nonzero(structure))
        all_weights_nonzero = nnz == structure.size
        if all_weights_nonzero:
            center_is_true = True
        else:
            center_is_true = _center_is_true(structure, origin)
    erode_kernel = _get_binary_erosion_kernel(structure.shape, int_type, offsets, center_is_true, border_value, invert, masked, all_weights_nonzero)
    if iterations == 1:
        if masked:
            output = erode_kernel(input, structure, mask, output)
        else:
            output = erode_kernel(input, structure, output)
    elif center_is_true and (not brute_force):
        raise NotImplementedError('only brute_force iteration has been implemented')
    else:
        if cupy.shares_memory(output, input, 'MAY_SHARE_BOUNDS'):
            raise ValueError('output and input may not overlap in memory')
        tmp_in = cupy.empty_like(input, dtype=output.dtype)
        tmp_out = output
        if iterations >= 1 and (not iterations & 1):
            tmp_in, tmp_out = (tmp_out, tmp_in)
        if masked:
            tmp_out = erode_kernel(input, structure, mask, tmp_out)
        else:
            tmp_out = erode_kernel(input, structure, tmp_out)
        changed = not (input == tmp_out).all()
        ii = 1
        while ii < iterations or (iterations < 1 and changed):
            tmp_in, tmp_out = (tmp_out, tmp_in)
            if masked:
                tmp_out = erode_kernel(tmp_in, structure, mask, tmp_out)
            else:
                tmp_out = erode_kernel(tmp_in, structure, tmp_out)
            changed = not (tmp_in == tmp_out).all()
            ii += 1
            if not changed and (not ii & 1):
                break
        output = tmp_out
    if temp_needed:
        _core.elementwise_copy(output, temp)
        output = temp
    return output