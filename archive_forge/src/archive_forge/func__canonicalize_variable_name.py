import re
def _canonicalize_variable_name(name):
    if name is None:
        return 'Variable'
    name = _VARIABLE_UNIQUIFYING_REGEX.sub('/', name)
    name = _VARIABLE_UNIQUIFYING_REGEX_AT_END.sub('', name)
    return name