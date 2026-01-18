from collections import abc
import copy
def traverse_obj(obj):
    oid = id(obj)
    if oid in visited or isinstance(obj, str):
        return
    visited.add(oid)
    if hasattr(obj, 'set_current_view_type'):
        obj.set_current_view_type(tp, visited=visited)
    if isinstance(obj, abc.Sequence):
        for item in obj:
            traverse_obj(item)
    elif isinstance(obj, abc.Mapping):
        for val in obj.values():
            traverse_obj(val)