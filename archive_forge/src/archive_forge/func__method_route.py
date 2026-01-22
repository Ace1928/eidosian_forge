from __future__ import annotations
import importlib.util
import os
import pathlib
import sys
import typing as t
from collections import defaultdict
from functools import update_wrapper
from jinja2 import BaseLoader
from jinja2 import FileSystemLoader
from werkzeug.exceptions import default_exceptions
from werkzeug.exceptions import HTTPException
from werkzeug.utils import cached_property
from .. import typing as ft
from ..helpers import get_root_path
from ..templating import _default_template_ctx_processor
def _method_route(self, method: str, rule: str, options: dict[str, t.Any]) -> t.Callable[[T_route], T_route]:
    if 'methods' in options:
        raise TypeError("Use the 'route' decorator to use the 'methods' argument.")
    return self.route(rule, methods=[method], **options)