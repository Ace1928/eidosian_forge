import mimetypes
from time import time as now
from email.utils import formatdate, parsedate_tz, mktime_tz
from urllib.request import AbstractDigestAuthHandler, parse_keqv_list, parse_http_list
from .httpexceptions import HTTPBadRequest
class EnvironVariable(str):
    """
    a CGI ``environ`` variable as described by WSGI

    This is a helper object so that standard WSGI ``environ`` variables
    can be extracted w/o syntax error possibility.
    """

    def __call__(self, environ):
        return environ.get(self, '')

    def __repr__(self):
        return '<EnvironVariable %s>' % self

    def update(self, environ, value):
        environ[self] = value