import json
from string import Template
import re
import sys
from webob.acceptparse import create_accept_header
from webob.compat import (
from webob.request import Request
from webob.response import Response
from webob.util import html_escape
class HTTPExceptionMiddleware(object):
    """
    Middleware that catches exceptions in the sub-application.  This
    does not catch exceptions in the app_iter; only during the initial
    calling of the application.

    This should be put *very close* to applications that might raise
    these exceptions.  This should not be applied globally; letting
    *expected* exceptions raise through the WSGI stack is dangerous.
    """

    def __init__(self, application):
        self.application = application

    def __call__(self, environ, start_response):
        try:
            return self.application(environ, start_response)
        except HTTPException:
            parent_exc_info = sys.exc_info()

            def repl_start_response(status, headers, exc_info=None):
                if exc_info is None:
                    exc_info = parent_exc_info
                return start_response(status, headers, exc_info)
            return parent_exc_info[1](environ, repl_start_response)