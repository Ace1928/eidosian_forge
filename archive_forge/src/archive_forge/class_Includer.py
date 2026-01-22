import warnings
from io import StringIO
class Includer(Recursive):
    """
    Starts another request with the given path and adding or
    overwriting any values in the `extra_environ` dictionary.
    Returns an IncludeResponse object.
    """

    def activate(self, environ):
        response = IncludedResponse()

        def start_response(status, headers, exc_info=None):
            if exc_info:
                raise exc_info
            response.status = status
            response.headers = headers
            return response.write
        app_iter = self.application(environ, start_response)
        try:
            for s in app_iter:
                response.write(s)
        finally:
            if hasattr(app_iter, 'close'):
                app_iter.close()
        response.close()
        return response