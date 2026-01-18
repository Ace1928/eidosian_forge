import re
import warnings
from pprint import pformat
from http.cookies import SimpleCookie
from paste.request import EnvironHeaders, get_cookie_dict, \
from paste.util.multidict import MultiDict, UnicodeMultiDict
from paste.registry import StackedObjectProxy
from paste.response import HeaderDict
from paste.wsgilib import encode_unicode_app_iter
from paste.httpheaders import ACCEPT_LANGUAGE
from paste.util.mimeparse import desired_matches

        Get/set the Content-Type header (or None), *without* the
        charset or any parameters.

        If you include parameters (or ``;`` at all) when setting the
        content_type, any existing parameters will be deleted;
        otherwise they will be preserved.
        