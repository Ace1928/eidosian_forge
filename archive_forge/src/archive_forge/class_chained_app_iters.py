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
class chained_app_iters(object):
    """
    Chains several app_iters together, also delegating .close() to each
    of them.
    """

    def __init__(self, *chained):
        self.app_iters = chained
        self.chained = [iter(item) for item in chained]
        self._closed = False

    def __iter__(self):
        return self

    def next(self):
        if len(self.chained) == 1:
            return next(self.chained[0])
        else:
            try:
                return next(self.chained[0])
            except StopIteration:
                self.chained.pop(0)
                return self.next()
    __next__ = next

    def close(self):
        self._closed = True
        got_exc = None
        for app_iter in self.app_iters:
            try:
                if hasattr(app_iter, 'close'):
                    app_iter.close()
            except:
                got_exc = sys.exc_info()
        if got_exc:
            raise got_exc

    def __del__(self):
        if not self._closed:
            print('Error: app_iter.close() was not called when finishing WSGI request. finalization function %s not called' % self.close_func, file=sys.stderr)