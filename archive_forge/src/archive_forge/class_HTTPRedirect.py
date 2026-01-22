import io
import contextlib
import urllib.parse
from sys import exc_info as _exc_info
from traceback import format_exception as _format_exception
from xml.sax import saxutils
import html
from more_itertools import always_iterable
import cherrypy
from cherrypy._cpcompat import ntob
from cherrypy._cpcompat import tonative
from cherrypy._helper import classproperty
from cherrypy.lib import httputil as _httputil
class HTTPRedirect(CherryPyException):
    """Exception raised when the request should be redirected.

    This exception will force a HTTP redirect to the URL or URL's you give it.
    The new URL must be passed as the first argument to the Exception,
    e.g., HTTPRedirect(newUrl). Multiple URLs are allowed in a list.
    If a URL is absolute, it will be used as-is. If it is relative, it is
    assumed to be relative to the current cherrypy.request.path_info.

    If one of the provided URL is a unicode object, it will be encoded
    using the default encoding or the one passed in parameter.

    There are multiple types of redirect, from which you can select via the
    ``status`` argument. If you do not provide a ``status`` arg, it defaults to
    303 (or 302 if responding with HTTP/1.0).

    Examples::

        raise cherrypy.HTTPRedirect("")
        raise cherrypy.HTTPRedirect("/abs/path", 307)
        raise cherrypy.HTTPRedirect(["path1", "path2?a=1&b=2"], 301)

    See :ref:`redirectingpost` for additional caveats.
    """
    urls = None
    "The list of URL's to emit."
    encoding = 'utf-8'
    'The encoding when passed urls are not native strings'

    def __init__(self, urls, status=None, encoding=None):
        self.urls = abs_urls = [urllib.parse.urljoin(cherrypy.url(), tonative(url, encoding or self.encoding)) for url in always_iterable(urls)]
        status = int(status) if status is not None else self.default_status
        if not 300 <= status <= 399:
            raise ValueError('status must be between 300 and 399.')
        CherryPyException.__init__(self, abs_urls, status)

    @classproperty
    def default_status(cls):
        """
        The default redirect status for the request.

        RFC 2616 indicates a 301 response code fits our goal; however,
        browser support for 301 is quite messy. Use 302/303 instead. See
        http://www.alanflavell.org.uk/www/post-redirect.html
        """
        return 303 if cherrypy.serving.request.protocol >= (1, 1) else 302

    @property
    def status(self):
        """The integer HTTP status code to emit."""
        _, status = self.args[:2]
        return status

    def set_response(self):
        """Modify cherrypy.response status, headers, and body to represent
        self.

        CherryPy uses this internally, but you can also use it to create an
        HTTPRedirect object and set its output without *raising* the exception.
        """
        response = cherrypy.serving.response
        response.status = status = self.status
        if status in (300, 301, 302, 303, 307, 308):
            response.headers['Content-Type'] = 'text/html;charset=utf-8'
            response.headers['Location'] = self.urls[0]
            msg = {300: 'This resource can be found at ', 301: 'This resource has permanently moved to ', 302: 'This resource resides temporarily at ', 303: 'This resource can be found at ', 307: 'This resource has moved temporarily to ', 308: 'This resource has been moved to '}[status]
            msg += '<a href=%s>%s</a>.'
            msgs = [msg % (saxutils.quoteattr(u), html.escape(u, quote=False)) for u in self.urls]
            response.body = ntob('<br />\n'.join(msgs), 'utf-8')
            response.headers.pop('Content-Length', None)
        elif status == 304:
            for key in ('Allow', 'Content-Encoding', 'Content-Language', 'Content-Length', 'Content-Location', 'Content-MD5', 'Content-Range', 'Content-Type', 'Expires', 'Last-Modified'):
                if key in response.headers:
                    del response.headers[key]
            response.body = None
            response.headers.pop('Content-Length', None)
        elif status == 305:
            response.headers['Location'] = ntob(self.urls[0], 'utf-8')
            response.body = None
            response.headers.pop('Content-Length', None)
        else:
            raise ValueError('The %s status code is unknown.' % status)

    def __call__(self):
        """Use this exception as a request.handler (raise self)."""
        raise self