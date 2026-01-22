import numpy as np
def _from_ctypes_union(t):
    import ctypes
    formats = []
    offsets = []
    names = []
    for fname, ftyp in t._fields_:
        names.append(fname)
        formats.append(dtype_from_ctypes_type(ftyp))
        offsets.append(0)
    return np.dtype(dict(formats=formats, offsets=offsets, names=names, itemsize=ctypes.sizeof(t)))