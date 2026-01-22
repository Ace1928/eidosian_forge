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
class add_close:
    """
    An an iterable that iterates over app_iter, then calls
    close_func.
    """

    def __init__(self, app_iterable, close_func):
        self.app_iterable = app_iterable
        self.app_iter = iter(app_iterable)
        self.close_func = close_func
        self._closed = False

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        return next(self.app_iter)

    def close(self):
        self._closed = True
        if hasattr(self.app_iterable, 'close'):
            self.app_iterable.close()
        self.close_func()

    def __del__(self):
        if not self._closed:
            print('Error: app_iter.close() was not called when finishing WSGI request. finalization function %s not called' % self.close_func, file=sys.stderr)