import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
def _encode_idna(self, url):
    global urlparse
    if urlparse is None:
        from urllib import parse as urlparse
    try:
        scheme, netloc, path, params, query, fragment = urlparse.urlparse(url)
    except ValueError:
        return url
    try:
        netloc = netloc.encode('idna')
        netloc = netloc.decode('ascii')
        return str(urlparse.urlunparse((scheme, netloc, path, params, query, fragment)))
    except UnicodeError:
        return url