from __future__ import annotations
import ast
import collections.abc as cabc
import importlib.metadata
import inspect
import os
import platform
import re
import sys
import traceback
import typing as t
from functools import update_wrapper
from operator import itemgetter
from types import ModuleType
import click
from click.core import ParameterSource
from werkzeug import run_simple
from werkzeug.serving import is_running_from_reloader
from werkzeug.utils import import_string
from .globals import current_app
from .helpers import get_debug_flag
from .helpers import get_load_dotenv
class ScriptInfo:
    """Helper object to deal with Flask applications.  This is usually not
    necessary to interface with as it's used internally in the dispatching
    to click.  In future versions of Flask this object will most likely play
    a bigger role.  Typically it's created automatically by the
    :class:`FlaskGroup` but you can also manually create it and pass it
    onwards as click object.
    """

    def __init__(self, app_import_path: str | None=None, create_app: t.Callable[..., Flask] | None=None, set_debug_flag: bool=True) -> None:
        self.app_import_path = app_import_path
        self.create_app = create_app
        self.data: dict[t.Any, t.Any] = {}
        self.set_debug_flag = set_debug_flag
        self._loaded_app: Flask | None = None

    def load_app(self) -> Flask:
        """Loads the Flask app (if not yet loaded) and returns it.  Calling
        this multiple times will just result in the already loaded app to
        be returned.
        """
        if self._loaded_app is not None:
            return self._loaded_app
        if self.create_app is not None:
            app: Flask | None = self.create_app()
        elif self.app_import_path:
            path, name = (re.split(':(?![\\\\/])', self.app_import_path, maxsplit=1) + [None])[:2]
            import_name = prepare_import(path)
            app = locate_app(import_name, name)
        else:
            for path in ('wsgi.py', 'app.py'):
                import_name = prepare_import(path)
                app = locate_app(import_name, None, raise_if_not_found=False)
                if app is not None:
                    break
        if app is None:
            raise NoAppException("Could not locate a Flask application. Use the 'flask --app' option, 'FLASK_APP' environment variable, or a 'wsgi.py' or 'app.py' file in the current directory.")
        if self.set_debug_flag:
            app.debug = get_debug_flag()
        self._loaded_app = app
        return app