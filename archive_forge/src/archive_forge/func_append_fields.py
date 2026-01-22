import itertools
import numpy as np
import numpy.ma as ma
from numpy import ndarray, recarray
from numpy.ma import MaskedArray
from numpy.ma.mrecords import MaskedRecords
from numpy.core.overrides import array_function_dispatch
from numpy.lib._iotools import _is_string_like
@array_function_dispatch(_append_fields_dispatcher)
def append_fields(base, names, data, dtypes=None, fill_value=-1, usemask=True, asrecarray=False):
    """
    Add new fields to an existing array.

    The names of the fields are given with the `names` arguments,
    the corresponding values with the `data` arguments.
    If a single field is appended, `names`, `data` and `dtypes` do not have
    to be lists but just values.

    Parameters
    ----------
    base : array
        Input array to extend.
    names : string, sequence
        String or sequence of strings corresponding to the names
        of the new fields.
    data : array or sequence of arrays
        Array or sequence of arrays storing the fields to add to the base.
    dtypes : sequence of datatypes, optional
        Datatype or sequence of datatypes.
        If None, the datatypes are estimated from the `data`.
    fill_value : {float}, optional
        Filling value used to pad missing data on the shorter arrays.
    usemask : {False, True}, optional
        Whether to return a masked array or not.
    asrecarray : {False, True}, optional
        Whether to return a recarray (MaskedRecords) or not.

    """
    if isinstance(names, (tuple, list)):
        if len(names) != len(data):
            msg = 'The number of arrays does not match the number of names'
            raise ValueError(msg)
    elif isinstance(names, str):
        names = [names]
        data = [data]
    if dtypes is None:
        data = [np.array(a, copy=False, subok=True) for a in data]
        data = [a.view([(name, a.dtype)]) for name, a in zip(names, data)]
    else:
        if not isinstance(dtypes, (tuple, list)):
            dtypes = [dtypes]
        if len(data) != len(dtypes):
            if len(dtypes) == 1:
                dtypes = dtypes * len(data)
            else:
                msg = 'The dtypes argument must be None, a dtype, or a list.'
                raise ValueError(msg)
        data = [np.array(a, copy=False, subok=True, dtype=d).view([(n, d)]) for a, n, d in zip(data, names, dtypes)]
    base = merge_arrays(base, usemask=usemask, fill_value=fill_value)
    if len(data) > 1:
        data = merge_arrays(data, flatten=True, usemask=usemask, fill_value=fill_value)
    else:
        data = data.pop()
    output = ma.masked_all(max(len(base), len(data)), dtype=_get_fieldspec(base.dtype) + _get_fieldspec(data.dtype))
    output = recursive_fill_fields(base, output)
    output = recursive_fill_fields(data, output)
    return _fix_output(output, usemask=usemask, asrecarray=asrecarray)