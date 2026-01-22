import numpy
import warnings
from numpy.lib.utils import safe_eval, drop_metadata
from numpy.compat import (
def _read_array_header(fp, version, max_header_size=_MAX_HEADER_SIZE):
    """
    see read_array_header_1_0
    """
    import struct
    hinfo = _header_size_info.get(version)
    if hinfo is None:
        raise ValueError('Invalid version {!r}'.format(version))
    hlength_type, encoding = hinfo
    hlength_str = _read_bytes(fp, struct.calcsize(hlength_type), 'array header length')
    header_length = struct.unpack(hlength_type, hlength_str)[0]
    header = _read_bytes(fp, header_length, 'array header')
    header = header.decode(encoding)
    if len(header) > max_header_size:
        raise ValueError(f'Header info length ({len(header)}) is large and may not be safe to load securely.\nTo allow loading, adjust `max_header_size` or fully trust the `.npy` file using `allow_pickle=True`.\nFor safety against large resource use or crashes, sandboxing may be necessary.')
    try:
        d = safe_eval(header)
    except SyntaxError as e:
        if version <= (2, 0):
            header = _filter_header(header)
            try:
                d = safe_eval(header)
            except SyntaxError as e2:
                msg = 'Cannot parse header: {!r}'
                raise ValueError(msg.format(header)) from e2
            else:
                warnings.warn('Reading `.npy` or `.npz` file required additional header parsing as it was created on Python 2. Save the file again to speed up loading and avoid this warning.', UserWarning, stacklevel=4)
        else:
            msg = 'Cannot parse header: {!r}'
            raise ValueError(msg.format(header)) from e
    if not isinstance(d, dict):
        msg = 'Header is not a dictionary: {!r}'
        raise ValueError(msg.format(d))
    if EXPECTED_KEYS != d.keys():
        keys = sorted(d.keys())
        msg = 'Header does not contain the correct keys: {!r}'
        raise ValueError(msg.format(keys))
    if not isinstance(d['shape'], tuple) or not all((isinstance(x, int) for x in d['shape'])):
        msg = 'shape is not valid: {!r}'
        raise ValueError(msg.format(d['shape']))
    if not isinstance(d['fortran_order'], bool):
        msg = 'fortran_order is not a valid bool: {!r}'
        raise ValueError(msg.format(d['fortran_order']))
    try:
        dtype = descr_to_dtype(d['descr'])
    except TypeError as e:
        msg = 'descr is not a valid dtype descriptor: {!r}'
        raise ValueError(msg.format(d['descr'])) from e
    return (d['shape'], d['fortran_order'], dtype)