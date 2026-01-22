import re
import numpy
import cupy
import cupy._core._routines_manipulation as _manipulation
from cupy._core._dtype import get_dtype, _raise_if_invalid_cast
from cupy._core import internal
def _validate_normalize_axes(axes, axis, keepdims, input_coredimss, output_coredimss):
    nin = len(input_coredimss)
    nout = 1 if not isinstance(output_coredimss, list) else len(output_coredimss)
    if axes is not None and axis is not None:
        raise ValueError('Only one of `axis` or `axes` keyword arguments should be given')
    if axes and (not isinstance(axes, list)):
        raise ValueError('`axes` has to be of type list')
    filtered_core_dims = list(filter(len, input_coredimss))
    nr_outputs_with_coredims = len([True for x in output_coredimss if len(x) > 0])
    if keepdims:
        if nr_outputs_with_coredims > 0:
            raise ValueError('`keepdims` can only be used for scalar outputs')
        output_coredimss = len(output_coredimss) * [filtered_core_dims[0]]
    core_dims = input_coredimss + output_coredimss
    if axis is not None:
        if not isinstance(axis, int):
            raise ValueError('`axis` argument has to be an integer value')
        if filtered_core_dims:
            cd0 = filtered_core_dims[0]
            if len(cd0) != 1:
                raise ValueError('`axis` can be used only, if one core dimension is present')
            for cd in filtered_core_dims:
                if cd0 != cd:
                    raise ValueError('To use `axis`, all core dimensions have to be equal')
    if axes is None:
        if axis is not None:
            axes = [(axis,) if cd else tuple() for cd in core_dims]
        else:
            axes = [tuple(range(-len(icd), 0)) for icd in core_dims]
    axes = [(a,) if isinstance(a, int) else a for a in axes]
    if nr_outputs_with_coredims == 0 and nin != len(axes) and (nin + nout != len(axes)) or (nr_outputs_with_coredims > 0 and nin + nout != len(axes)):
        raise ValueError('The number of `axes` entries is not equal the number of input and output arguments')
    output_axes = axes[nin:]
    output_axes = output_axes if output_axes else [tuple(range(-len(ocd), 0)) for ocd in output_coredimss]
    input_axes = axes[:nin]
    for idx, (iax, icd) in enumerate(zip(input_axes, input_coredimss)):
        if len(iax) != len(icd):
            raise ValueError(f'The number of `axes` entries for argument #{idx} is not equal the number of respective input core dimensions in signature')
    if not keepdims:
        for idx, (oax, ocd) in enumerate(zip(output_axes, output_coredimss)):
            if len(oax) != len(ocd):
                raise ValueError(f'The number of `axes` entries for argument #{idx} is not equal the number of respective output core dimensions in signature')
    elif input_coredimss:
        icd0 = input_coredimss[0]
        for icd in input_coredimss:
            if icd0 != icd:
                raise ValueError('To use `keepdims`, all core dimensions have to be equal')
        iax0 = input_axes[0]
        output_axes = [iax0 for _ in output_coredimss]
    return (input_axes, output_axes)