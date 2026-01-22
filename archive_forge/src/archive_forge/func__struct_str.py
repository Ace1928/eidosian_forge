import numpy as np
def _struct_str(dtype, include_align):
    if not (include_align and dtype.isalignedstruct) and _is_packed(dtype):
        sub = _struct_list_str(dtype)
    else:
        sub = _struct_dict_str(dtype, include_align)
    if dtype.type != np.void:
        return '({t.__module__}.{t.__name__}, {f})'.format(t=dtype.type, f=sub)
    else:
        return sub