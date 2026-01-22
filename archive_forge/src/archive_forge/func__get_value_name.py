import re
def _get_value_name(mod, value, pattern):
    for k, v in mod.__dict__.items():
        if k.startswith(pattern):
            if v == value:
                return k
    return 'Unknown'