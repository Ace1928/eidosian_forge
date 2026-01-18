from __future__ import annotations
import logging
import sys
import typing as t
from werkzeug.local import LocalProxy
from .globals import request
@LocalProxy
def wsgi_errors_stream() -> t.TextIO:
    """Find the most appropriate error stream for the application. If a request
    is active, log to ``wsgi.errors``, otherwise use ``sys.stderr``.

    If you configure your own :class:`logging.StreamHandler`, you may want to
    use this for the stream. If you are using file or dict configuration and
    can't import this directly, you can refer to it as
    ``ext://flask.logging.wsgi_errors_stream``.
    """
    if request:
        return request.environ['wsgi.errors']
    return sys.stderr