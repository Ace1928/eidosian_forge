import html
import sys
import os
import traceback
from io import StringIO
import pprint
import itertools
import time
import re
from paste.exceptions import errormiddleware, formatter, collector
from paste import wsgilib
from paste import urlparser
from paste import httpexceptions
from paste import registry
from paste import request
from paste import response
from paste.evalexception import evalcontext
def wsgiapp_wrapper(*args):
    if len(args) == 3:
        environ = args[1]
        start_response = args[2]
        args = [args[0]]
    else:
        environ, start_response = args
        args = []

    def application(environ, start_response):
        form = wsgilib.parse_formvars(environ, include_get_vars=True)
        headers = response.HeaderDict({'content-type': 'text/html', 'status': '200 OK'})
        form['environ'] = environ
        form['headers'] = headers
        res = func(*args, **form.mixed())
        status = headers.pop('status')
        start_response(status, headers.headeritems())
        return [res]
    app = httpexceptions.make_middleware(application)
    app = simplecatcher(app)
    return app(environ, start_response)