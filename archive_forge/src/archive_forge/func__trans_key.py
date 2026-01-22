from webob.compat import (
from webob.multidict import MultiDict
def _trans_key(key):
    if not isinstance(key, string_types):
        return None
    elif key in key2header:
        return key2header[key]
    elif key.startswith('HTTP_'):
        return key[5:].replace('_', '-').title()
    else:
        return None