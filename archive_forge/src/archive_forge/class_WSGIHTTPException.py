import json
from string import Template
import re
import sys
from webob.acceptparse import create_accept_header
from webob.compat import (
from webob.request import Request
from webob.response import Response
from webob.util import html_escape
class WSGIHTTPException(Response, HTTPException):
    code = 500
    title = 'Internal Server Error'
    explanation = ''
    body_template_obj = Template('${explanation}<br /><br />\n${detail}\n${html_comment}\n')
    plain_template_obj = Template('${status}\n\n${body}')
    html_template_obj = Template('<html>\n <head>\n  <title>${status}</title>\n </head>\n <body>\n  <h1>${status}</h1>\n  ${body}\n </body>\n</html>')
    empty_body = False

    def __init__(self, detail=None, headers=None, comment=None, body_template=None, json_formatter=None, **kw):
        Response.__init__(self, status='%s %s' % (self.code, self.title), **kw)
        Exception.__init__(self, detail)
        if headers:
            self.headers.extend(headers)
        self.detail = detail
        self.comment = comment
        if body_template is not None:
            self.body_template = body_template
            self.body_template_obj = Template(body_template)
        if self.empty_body:
            del self.content_type
            del self.content_length
        if json_formatter is not None:
            self.json_formatter = json_formatter

    def __str__(self):
        return self.detail or self.explanation

    def _make_body(self, environ, escape):
        escape = lazify(escape)
        args = {'explanation': escape(self.explanation), 'detail': escape(self.detail or ''), 'comment': escape(self.comment or '')}
        if self.comment:
            args['html_comment'] = '<!-- %s -->' % escape(self.comment)
        else:
            args['html_comment'] = ''
        if WSGIHTTPException.body_template_obj is not self.body_template_obj:
            for k, v in environ.items():
                args[k] = escape(v)
            for k, v in self.headers.items():
                args[k.lower()] = escape(v)
        t_obj = self.body_template_obj
        return t_obj.safe_substitute(args)

    def plain_body(self, environ):
        body = self._make_body(environ, no_escape)
        body = strip_tags(body)
        return self.plain_template_obj.substitute(status=self.status, title=self.title, body=body)

    def html_body(self, environ):
        body = self._make_body(environ, html_escape)
        return self.html_template_obj.substitute(status=self.status, body=body)

    def json_formatter(self, body, status, title, environ):
        return {'message': body, 'code': status, 'title': title}

    def json_body(self, environ):
        body = self._make_body(environ, no_escape)
        jsonbody = self.json_formatter(body=body, status=self.status, title=self.title, environ=environ)
        return json.dumps(jsonbody)

    def generate_response(self, environ, start_response):
        if self.content_length is not None:
            del self.content_length
        headerlist = list(self.headerlist)
        accept_value = environ.get('HTTP_ACCEPT', '')
        accept_header = create_accept_header(header_value=accept_value)
        acceptable_offers = accept_header.acceptable_offers(offers=['text/html', 'application/json'])
        match = acceptable_offers[0][0] if acceptable_offers else None
        if match == 'text/html':
            content_type = 'text/html'
            body = self.html_body(environ)
        elif match == 'application/json':
            content_type = 'application/json'
            body = self.json_body(environ)
        else:
            content_type = 'text/plain'
            body = self.plain_body(environ)
        resp = Response(body, status=self.status, headerlist=headerlist, content_type=content_type)
        resp.content_type = content_type
        return resp(environ, start_response)

    def __call__(self, environ, start_response):
        is_head = environ['REQUEST_METHOD'] == 'HEAD'
        if self.has_body or self.empty_body or is_head:
            app_iter = Response.__call__(self, environ, start_response)
        else:
            app_iter = self.generate_response(environ, start_response)
        if is_head:
            app_iter = []
        return app_iter

    @property
    def wsgi_response(self):
        return self