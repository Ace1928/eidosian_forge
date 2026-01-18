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
def urlvars(self):
    """
        Return any variables matched in the URL (e.g.,
        ``wsgiorg.routing_args``).
        """
    if 'paste.urlvars' in self.environ:
        return self.environ['paste.urlvars']
    elif 'wsgiorg.routing_args' in self.environ:
        return self.environ['wsgiorg.routing_args'][1]
    else:
        return {}