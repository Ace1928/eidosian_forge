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
def _preparse_for_stopping_flags(application_klass, argv):
    """Looks for 'help', 'version', and 'generate-config; commands
    in command line. If found, raises the help and version of
    current Application.

    This is useful for traitlets applications that have to parse
    the command line multiple times, but want to control when
    when 'help' and 'version' is raised.
    """
    try:
        interpreted_argv = argv[:argv.index('--')]
    except ValueError:
        interpreted_argv = argv
    if any((x in interpreted_argv for x in ('-h', '--help-all', '--help'))):
        app = application_klass()
        app.print_help('--help-all' in interpreted_argv)
        app.exit(0)
    if '--version' in interpreted_argv or '-V' in interpreted_argv:
        app = application_klass()
        app.print_version()
        app.exit(0)
    if '--generate-config' in interpreted_argv:
        app = application_klass()
        app.write_default_config()
        app.exit(0)