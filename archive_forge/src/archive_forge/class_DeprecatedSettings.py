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
class DeprecatedSettings(StackedObjectProxy):

    def _push_object(self, obj):
        warnings.warn('paste.wsgiwrappers.settings is deprecated: Please use paste.wsgiwrappers.WSGIRequest.defaults instead', DeprecationWarning, 3)
        WSGIResponse.defaults._push_object(obj)
        StackedObjectProxy._push_object(self, obj)