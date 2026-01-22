import time
import functools
from hashlib import md5
from urllib.request import parse_http_list, parse_keqv_list
import cherrypy
from cherrypy._cpcompat import ntob, tonative
def _try_decode_header(header, charset):
    global FALLBACK_CHARSET
    for enc in (charset, FALLBACK_CHARSET):
        try:
            return tonative(ntob(tonative(header, 'latin1'), 'latin1'), enc)
        except ValueError as ve:
            last_err = ve
    else:
        raise last_err