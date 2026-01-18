from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING
from tornado.httputil import HTTPServerRequest
from tornado.web import HTTPError, authenticated
from bokeh.util.token import (
from .auth_request_handler import AuthRequestHandler
 Implements a custom Tornado handler for document display page

    