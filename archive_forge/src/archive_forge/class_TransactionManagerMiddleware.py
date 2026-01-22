from paste.httpexceptions import HTTPException
from wsgilib import catch_errors
class TransactionManagerMiddleware(object):

    def __init__(self, application):
        self.application = application

    def __call__(self, environ, start_response):
        environ['paste.transaction_manager'] = manager = Manager()
        environ['paste.throw_errors'] = True
        return catch_errors(self.application, environ, start_response, error_callback=manager.error, ok_callback=manager.finish)