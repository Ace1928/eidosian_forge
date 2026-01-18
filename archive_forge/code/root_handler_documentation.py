from __future__ import annotations
import logging # isort:skip
from tornado.web import authenticated
from .auth_request_handler import AuthRequestHandler
 Implements a custom Tornado handler to display the available applications
    If only one application it redirects to that application route
    