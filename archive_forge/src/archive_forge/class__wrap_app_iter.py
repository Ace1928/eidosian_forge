import io
import sys
import warnings
from traceback import print_exception
from io import StringIO
from urllib.parse import unquote, urlsplit
from paste.request import get_cookies, parse_querystring, parse_formvars
from paste.request import construct_url, path_info_split, path_info_pop
from paste.response import HeaderDict, has_header, header_value, remove_header
from paste.response import error_body_response, error_response, error_response_app
class _wrap_app_iter(object):

    def __init__(self, app_iterable, error_callback, ok_callback):
        self.app_iterable = app_iterable
        self.app_iter = iter(app_iterable)
        self.error_callback = error_callback
        self.ok_callback = ok_callback
        if hasattr(self.app_iterable, 'close'):
            self.close = self.app_iterable.close

    def __iter__(self):
        return self

    def next(self):
        try:
            return next(self.app_iter)
        except StopIteration:
            if self.ok_callback:
                self.ok_callback()
            raise
        except:
            self.error_callback(sys.exc_info())
            raise
    __next__ = next