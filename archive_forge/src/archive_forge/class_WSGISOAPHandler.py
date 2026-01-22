from __future__ import unicode_literals
import sys
import datetime
import sys
import logging
import warnings
import re
import traceback
from . import __author__, __copyright__, __license__, __version__
from .simplexml import SimpleXMLElement, TYPE_MAP, Date, Decimal
class WSGISOAPHandler(object):

    def __init__(self, dispatcher):
        self.dispatcher = dispatcher

    def __call__(self, environ, start_response):
        return self.handler(environ, start_response)

    def handler(self, environ, start_response):
        if environ['REQUEST_METHOD'] == 'GET':
            return self.do_get(environ, start_response)
        elif environ['REQUEST_METHOD'] == 'POST':
            return self.do_post(environ, start_response)
        else:
            start_response('405 Method not allowed', [('Content-Type', 'text/plain')])
            return ['Method not allowed']

    def do_get(self, environ, start_response):
        path = environ.get('PATH_INFO').lstrip('/')
        query = environ.get('QUERY_STRING')
        if path != '' and path not in self.dispatcher.methods.keys():
            start_response('404 Not Found', [('Content-Type', 'text/plain')])
            return ['Method not found: %s' % path]
        elif path == '':
            response = self.dispatcher.wsdl()
        else:
            req, res, doc = self.dispatcher.help(path)
            if len(query) == 0 or query == 'request':
                response = req
            else:
                response = res
        start_response('200 OK', [('Content-Type', 'text/xml'), ('Content-Length', str(len(response)))])
        return [response]

    def do_post(self, environ, start_response):
        length = int(environ['CONTENT_LENGTH'])
        request = environ['wsgi.input'].read(length)
        response = self.dispatcher.dispatch(request)
        start_response('200 OK', [('Content-Type', 'text/xml'), ('Content-Length', str(len(response)))])
        return [response]