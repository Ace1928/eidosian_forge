
    Used to signal that a request should be forwarded to a different location.

    ``url``
        The URL to forward to starting with a ``/`` and relative to
        ``RecursiveMiddleware``. URL fragments can also contain query strings
        so ``/error?code=404`` would be a valid URL fragment.

    ``environ``
        An altertative WSGI environment dictionary to use for the forwarded
        request. If specified is used *instead* of the ``url_fragment``

    ``factory``
        If specifed ``factory`` is used instead of ``url`` or ``environ``.
        ``factory`` is a callable that takes a WSGI application object
        as the first argument and returns an initialised WSGI middleware
        which can alter the forwarded response.

    Basic usage (must have ``RecursiveMiddleware`` present) :

    .. code-block:: python

        from pecan.middleware.recursive import ForwardRequestException
        def app(environ, start_response):
            if environ['PATH_INFO'] == '/hello':
                start_response("200 OK", [('Content-type', 'text/plain')])
                return ['Hello World!']
            elif environ['PATH_INFO'] == '/error':
                start_response("404 Not Found",
                    [('Content-type', 'text/plain')]
                )
                return ['Page not found']
            else:
                raise ForwardRequestException('/error')

        from pecan.middleware.recursive import RecursiveMiddleware
        app = RecursiveMiddleware(app)

    If you ran this application and visited ``/hello`` you would get a
    ``Hello World!`` message. If you ran the application and visited
    ``/not_found`` a ``ForwardRequestException`` would be raised and the caught
    by the ``RecursiveMiddleware``. The ``RecursiveMiddleware`` would then
    return the headers and response from the ``/error`` URL but would display
    a ``404 Not found`` status message.

    You could also specify an ``environ`` dictionary instead of a url. Using
    the same example as before:

    .. code-block:: python

        def app(environ, start_response):
            ... same as previous example ...
            else:
                new_environ = environ.copy()
                new_environ['PATH_INFO'] = '/error'
                raise ForwardRequestException(environ=new_environ)
    