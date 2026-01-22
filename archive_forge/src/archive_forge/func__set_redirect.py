import collections
import importlib
import os
import re
import sys
from fnmatch import fnmatch
from pathlib import Path
from os.path import isfile, join
from urllib.parse import parse_qs
import flask
from . import _validate
from ._utils import AttributeDict
from ._get_paths import get_relative_path
from ._callback_context import context_value
from ._get_app import get_app
def _set_redirect(redirect_from, path):
    app = get_app()
    if redirect_from and len(redirect_from):
        for redirect in redirect_from:
            fullname = app.get_relative_path(redirect)
            app.server.add_url_rule(fullname, fullname, _create_redirect_function(app.get_relative_path(path)))