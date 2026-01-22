from __future__ import annotations
import logging # isort:skip
import importlib.util
from os.path import isfile
from types import ModuleType
from typing import (
from tornado.httputil import HTTPServerRequest
from tornado.web import RequestHandler
from ..util.serialization import make_globally_unique_id
 A URL to redirect authenticated users to for logout.

        This proprty may return None.

        