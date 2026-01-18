import os
import tornado.web
from ..utils import strtobool
from ..views import BaseHandler
def write_error(self, status_code, **kwargs):
    exc_info = kwargs.get('exc_info')
    log_message = exc_info[1].log_message
    if log_message:
        self.write(log_message)
    self.set_status(status_code)
    self.finish()