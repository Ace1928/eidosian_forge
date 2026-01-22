import sys
import traceback
import cgi
from io import StringIO
from paste.exceptions import formatter, collector, reporter
from paste import wsgilib
from paste import request
class CatchingIter(object):
    """
    A wrapper around the application iterator that will catch
    exceptions raised by the a generator, or by the close method, and
    display or report as necessary.
    """

    def __init__(self, app_iter, environ, start_checker, error_middleware):
        self.app_iterable = app_iter
        self.app_iterator = iter(app_iter)
        self.environ = environ
        self.start_checker = start_checker
        self.error_middleware = error_middleware
        self.closed = False

    def __iter__(self):
        return self

    def next(self):
        __traceback_supplement__ = (Supplement, self.error_middleware, self.environ)
        if self.closed:
            raise StopIteration
        try:
            return next(self.app_iterator)
        except StopIteration:
            self.closed = True
            close_response = self._close()
            if close_response is not None:
                return close_response
            else:
                raise StopIteration
        except:
            self.closed = True
            close_response = self._close()
            exc_info = sys.exc_info()
            response = self.error_middleware.exception_handler(exc_info, self.environ)
            if close_response is not None:
                response += '<hr noshade>Error in .close():<br>%s' % close_response
            if not self.start_checker.response_started:
                self.start_checker('500 Internal Server Error', [('content-type', 'text/html')], exc_info)
            response = response.encode('utf8')
            return response
    __next__ = next

    def close(self):
        if not self.closed:
            self._close()

    def _close(self):
        """Close and return any error message"""
        if not hasattr(self.app_iterable, 'close'):
            return None
        try:
            self.app_iterable.close()
            return None
        except:
            close_response = self.error_middleware.exception_handler(sys.exc_info(), self.environ)
            return close_response