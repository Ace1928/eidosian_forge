import os
class _OFFSET_UNION(Union):
    _anonymous_ = ['_offset']
    _fields_ = [('_offset', _OFFSET), ('Pointer', PVOID)]