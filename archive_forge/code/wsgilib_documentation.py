import io
import sys
import warnings
from traceback import print_exception
from io import StringIO
from urllib.parse import unquote, urlsplit
from paste.request import get_cookies, parse_querystring, parse_formvars
from paste.request import construct_url, path_info_split, path_info_pop
from paste.response import HeaderDict, has_header, header_value, remove_header
from paste.response import error_body_response, error_response, error_response_app

    Runs application with environ and captures status, headers, and
    body.  None are sent on; you must send them on yourself (unlike
    ``capture_output``)

    Typically this is used like:

    .. code-block:: python

        def dehtmlifying_middleware(application):
            def replacement_app(environ, start_response):
                status, headers, body = intercept_output(
                    environ, application)
                start_response(status, headers)
                content_type = header_value(headers, 'content-type')
                if (not content_type
                    or not content_type.startswith('text/html')):
                    return [body]
                body = re.sub(r'<.*?>', '', body)
                return [body]
            return replacement_app

    A third optional argument ``conditional`` should be a function
    that takes ``conditional(status, headers)`` and returns False if
    the request should not be intercepted.  In that case
    ``start_response`` will be called and ``(None, None, app_iter)``
    will be returned.  You must detect that in your code and return
    the app_iter, like:

    .. code-block:: python

        def dehtmlifying_middleware(application):
            def replacement_app(environ, start_response):
                status, headers, body = intercept_output(
                    environ, application,
                    lambda s, h: header_value(headers, 'content-type').startswith('text/html'),
                    start_response)
                if status is None:
                    return body
                start_response(status, headers)
                body = re.sub(r'<.*?>', '', body)
                return [body]
            return replacement_app
    