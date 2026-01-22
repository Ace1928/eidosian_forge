import numpy as np
def _from_ctypes_structure(t):
    for item in t._fields_:
        if len(item) > 2:
            raise TypeError('ctypes bitfields have no dtype equivalent')
    if hasattr(t, '_pack_'):
        import ctypes
        formats = []
        offsets = []
        names = []
        current_offset = 0
        for fname, ftyp in t._fields_:
            names.append(fname)
            formats.append(dtype_from_ctypes_type(ftyp))
            effective_pack = min(t._pack_, ctypes.alignment(ftyp))
            current_offset = (current_offset + effective_pack - 1) // effective_pack * effective_pack
            offsets.append(current_offset)
            current_offset += ctypes.sizeof(ftyp)
        return np.dtype(dict(formats=formats, offsets=offsets, names=names, itemsize=ctypes.sizeof(t)))
    else:
        fields = []
        for fname, ftyp in t._fields_:
            fields.append((fname, dtype_from_ctypes_type(ftyp)))
        return np.dtype(fields, align=True)