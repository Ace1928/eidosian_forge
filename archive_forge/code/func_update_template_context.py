from __future__ import annotations
import collections.abc as cabc
import os
import sys
import typing as t
import weakref
from datetime import timedelta
from inspect import iscoroutinefunction
from itertools import chain
from types import TracebackType
from urllib.parse import quote as _url_quote
import click
from werkzeug.datastructures import Headers
from werkzeug.datastructures import ImmutableDict
from werkzeug.exceptions import BadRequestKeyError
from werkzeug.exceptions import HTTPException
from werkzeug.exceptions import InternalServerError
from werkzeug.routing import BuildError
from werkzeug.routing import MapAdapter
from werkzeug.routing import RequestRedirect
from werkzeug.routing import RoutingException
from werkzeug.routing import Rule
from werkzeug.serving import is_running_from_reloader
from werkzeug.wrappers import Response as BaseResponse
from . import cli
from . import typing as ft
from .ctx import AppContext
from .ctx import RequestContext
from .globals import _cv_app
from .globals import _cv_request
from .globals import current_app
from .globals import g
from .globals import request
from .globals import request_ctx
from .globals import session
from .helpers import get_debug_flag
from .helpers import get_flashed_messages
from .helpers import get_load_dotenv
from .helpers import send_from_directory
from .sansio.app import App
from .sansio.scaffold import _sentinel
from .sessions import SecureCookieSessionInterface
from .sessions import SessionInterface
from .signals import appcontext_tearing_down
from .signals import got_request_exception
from .signals import request_finished
from .signals import request_started
from .signals import request_tearing_down
from .templating import Environment
from .wrappers import Request
from .wrappers import Response
def update_template_context(self, context: dict[str, t.Any]) -> None:
    """Update the template context with some commonly used variables.
        This injects request, session, config and g into the template
        context as well as everything template context processors want
        to inject.  Note that the as of Flask 0.6, the original values
        in the context will not be overridden if a context processor
        decides to return a value with the same key.

        :param context: the context as a dictionary that is updated in place
                        to add extra variables.
        """
    names: t.Iterable[str | None] = (None,)
    if request:
        names = chain(names, reversed(request.blueprints))
    orig_ctx = context.copy()
    for name in names:
        if name in self.template_context_processors:
            for func in self.template_context_processors[name]:
                context.update(self.ensure_sync(func)())
    context.update(orig_ctx)