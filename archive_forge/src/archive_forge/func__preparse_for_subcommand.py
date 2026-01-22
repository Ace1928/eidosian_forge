from __future__ import annotations
import logging
import re
import sys
import typing as t
from jinja2 import Environment, FileSystemLoader
from jupyter_core.application import JupyterApp, NoStart
from tornado.log import LogFormatter
from tornado.web import RedirectHandler
from traitlets import Any, Bool, Dict, HasTraits, List, Unicode, default
from traitlets.config import Config
from jupyter_server.serverapp import ServerApp
from jupyter_server.transutils import _i18n
from jupyter_server.utils import is_namespace_package, url_path_join
from .handler import ExtensionHandlerMixin
def _preparse_for_subcommand(application_klass, argv):
    """Preparse command line to look for subcommands."""
    if len(argv) == 0:
        return None
    if application_klass.subcommands and len(argv) > 0:
        subc, subargv = (argv[0], argv[1:])
        if re.match('^\\w(\\-?\\w)*$', subc) and subc in application_klass.subcommands:
            app = application_klass()
            app.initialize_subcommand(subc, subargv)
            return app.subapp