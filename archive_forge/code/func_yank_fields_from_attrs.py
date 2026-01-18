import inspect
from functools import partial
from ..utils.module_loading import import_string
from .mountedtype import MountedType
from .unmountedtype import UnmountedType
def yank_fields_from_attrs(attrs, _as=None, sort=True):
    """
    Extract all the fields in given attributes (dict)
    and return them ordered
    """
    fields_with_names = []
    for attname, value in list(attrs.items()):
        field = get_field_as(value, _as)
        if not field:
            continue
        fields_with_names.append((attname, field))
    if sort:
        fields_with_names = sorted(fields_with_names, key=lambda f: f[1])
    return dict(fields_with_names)