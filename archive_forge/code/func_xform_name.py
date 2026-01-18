import logging
import os
import re
def xform_name(name, sep='_', _xform_cache=_xform_cache):
    """Convert camel case to a "pythonic" name.

    If the name contains the ``sep`` character, then it is
    returned unchanged.

    """
    if sep in name:
        return name
    key = (name, sep)
    if key not in _xform_cache:
        if _special_case_transform.search(name) is not None:
            is_special = _special_case_transform.search(name)
            matched = is_special.group()
            name = f'{name[:-len(matched)]}{sep}{matched.lower()}'
        s1 = _first_cap_regex.sub('\\1' + sep + '\\2', name)
        transformed = _end_cap_regex.sub('\\1' + sep + '\\2', s1).lower()
        _xform_cache[key] = transformed
    return _xform_cache[key]