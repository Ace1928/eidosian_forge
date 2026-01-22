import time
class SimpleApplication(object):
    """
    Produces a simple web page
    """

    def __call__(self, environ, start_response):
        body = b'<html><body>simple</body></html>'
        start_response('200 OK', [('Content-Type', 'text/html'), ('Content-Length', str(len(body)))])
        return [body]