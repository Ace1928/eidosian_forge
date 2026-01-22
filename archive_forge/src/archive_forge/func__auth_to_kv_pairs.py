from paste.httpexceptions import HTTPUnauthorized
from paste.httpheaders import (
import time, random
from urllib.parse import quote as url_quote
def _auth_to_kv_pairs(auth_string):
    """ split a digest auth string into key, value pairs """
    for item in _split_auth_string(auth_string):
        k, v = item.split('=', 1)
        if v.startswith('"') and len(v) > 1 and v.endswith('"'):
            v = v[1:-1]
        yield (k, v)