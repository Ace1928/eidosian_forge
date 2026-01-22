import sys
from .recursive import ForwardRequestException, RecursionLoop
class ErrorDocumentMiddleware(object):
    """
    Intersects HTTP response status code, looks it up in the error map defined
    in the Pecan app config.py, and routes to the controller assigned to that
    status.
    """

    def __init__(self, app, error_map):
        self.app = app
        self.error_map = error_map

    def __call__(self, environ, start_response):

        def replacement_start_response(status, headers, exc_info=None):
            """
            Overrides the default response if the status is defined in the
            Pecan app error map configuration.
            """
            try:
                status_code = int(status.split(' ')[0])
            except (ValueError, TypeError):
                raise Exception('ErrorDocumentMiddleware received an invalid status %s' % status)
            if status_code in self.error_map:

                def factory(app):
                    return StatusPersist(app, status, self.error_map[status_code])
                raise ForwardRequestException(factory=factory)
            return start_response(status, headers, exc_info)
        app_iter = self.app(environ, replacement_start_response)
        return app_iter