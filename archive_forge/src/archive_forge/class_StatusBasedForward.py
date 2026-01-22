import warnings
import sys
from urllib import parse as urlparse
from paste.recursive import ForwardRequestException, RecursiveMiddleware, RecursionLoop
from paste.util import converters
from paste.response import replace_header
class StatusBasedForward(object):
    """
    Middleware that lets you test a response against a custom mapper object to
    programatically determine whether to internally forward to another URL and
    if so, which URL to forward to.

    If you don't need the full power of this middleware you might choose to use
    the simpler ``forward`` middleware instead.

    The arguments are:

    ``app``
        The WSGI application or middleware chain.

    ``mapper``
        A callable that takes a status code as the
        first parameter, a message as the second, and accepts optional environ,
        global_conf and named argments afterwards. It should return a
        URL to forward to or ``None`` if the code is not to be intercepted.

    ``global_conf``
        Optional default configuration from your config file. If ``debug`` is
        set to ``true`` a message will be written to ``wsgi.errors`` on each
        internal forward stating the URL forwarded to.

    ``**params``
        Optional, any other configuration and extra arguments you wish to
        pass which will in turn be passed back to the custom mapper object.

    Here is an example where a ``404 File Not Found`` status response would be
    redirected to the URL ``/error?code=404&message=File%20Not%20Found``. This
    could be useful for passing the status code and message into another
    application to display an error document:

    .. code-block:: python

        from paste.errordocument import StatusBasedForward
        from paste.recursive import RecursiveMiddleware
        from urllib import urlencode

        def error_mapper(code, message, environ, global_conf, kw)
            if code in [404, 500]:
                params = urlencode({'message':message, 'code':code})
                url = '/error?'%(params)
                return url
            else:
                return None

        app = RecursiveMiddleware(
            StatusBasedForward(app, mapper=error_mapper),
        )

    """

    def __init__(self, app, mapper, global_conf=None, **params):
        if global_conf is None:
            global_conf = {}
        if global_conf:
            self.debug = converters.asbool(global_conf.get('debug', False))
        else:
            self.debug = False
        self.application = app
        self.mapper = mapper
        self.global_conf = global_conf
        self.params = params

    def __call__(self, environ, start_response):
        url = []

        def change_response(status, headers, exc_info=None):
            status_code = status.split(' ')
            try:
                code = int(status_code[0])
            except (ValueError, TypeError):
                raise Exception('StatusBasedForward middleware received an invalid status code %s' % repr(status_code[0]))
            message = ' '.join(status_code[1:])
            new_url = self.mapper(code, message, environ, self.global_conf, **self.params)
            if not (new_url == None or isinstance(new_url, str)):
                raise TypeError('Expected the url to internally redirect to in the StatusBasedForward mapperto be a string or None, not %r' % new_url)
            if new_url:
                url.append([new_url, status, headers])
                return [].append
            else:
                return start_response(status, headers, exc_info)
        app_iter = self.application(environ, change_response)
        if url:
            if hasattr(app_iter, 'close'):
                app_iter.close()

            def factory(app):
                return StatusKeeper(app, status=url[0][1], url=url[0][0], headers=url[0][2])
            raise ForwardRequestException(factory=factory)
        else:
            return app_iter