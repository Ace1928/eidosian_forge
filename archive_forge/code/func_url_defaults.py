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
@setupmethod
def url_defaults(self, f: T_url_defaults) -> T_url_defaults:
    """Callback function for URL defaults for all view functions of the
        application.  It's called with the endpoint and values and should
        update the values passed in place.

        This is available on both app and blueprint objects. When used on an app, this
        is called for every request. When used on a blueprint, this is called for
        requests that the blueprint handles. To register with a blueprint and affect
        every request, use :meth:`.Blueprint.app_url_defaults`.
        """
    self.url_default_functions[None].append(f)
    return f