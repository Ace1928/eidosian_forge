import json
from string import Template
import re
import sys
from webob.acceptparse import create_accept_header
from webob.compat import (
from webob.request import Request
from webob.response import Response
from webob.util import html_escape
class _HTTPMove(HTTPRedirection):
    """
    redirections which require a Location field

    Since a 'Location' header is a required attribute of 301, 302, 303,
    305, 307 and 308 (but not 304), this base class provides the mechanics to
    make this easy.

    You can provide a location keyword argument to set the location
    immediately.  You may also give ``add_slash=True`` if you want to
    redirect to the same URL as the request, except with a ``/`` added
    to the end.

    Relative URLs in the location will be resolved to absolute.
    """
    explanation = 'The resource has been moved to'
    body_template_obj = Template('${explanation} <a href="${location}">${location}</a>;\nyou should be redirected automatically.\n${detail}\n${html_comment}')

    def __init__(self, detail=None, headers=None, comment=None, body_template=None, location=None, add_slash=False):
        super(_HTTPMove, self).__init__(detail=detail, headers=headers, comment=comment, body_template=body_template)
        if location is not None:
            if '\n' in location or '\r' in location:
                raise ValueError('Control characters are not allowed in location')
            self.location = location
            if add_slash:
                raise TypeError('You can only provide one of the arguments location and add_slash')
        self.add_slash = add_slash

    def __call__(self, environ, start_response):
        req = Request(environ)
        if self.add_slash:
            url = req.path_url
            url += '/'
            if req.environ.get('QUERY_STRING'):
                url += '?' + req.environ['QUERY_STRING']
            self.location = url
        self.location = urlparse.urljoin(req.path_url, self.location)
        return super(_HTTPMove, self).__call__(environ, start_response)