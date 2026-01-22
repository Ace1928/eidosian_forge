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
def _link_jupyter_server_extension(self, serverapp: ServerApp) -> None:
    """Link the ExtensionApp to an initialized ServerApp.

        The ServerApp is stored as an attribute and config
        is exchanged between ServerApp and `self` in case
        the command line contains traits for the ExtensionApp
        or the ExtensionApp's config files have server
        settings.

        Note, the ServerApp has not initialized the Tornado
        Web Application yet, so do not try to affect the
        `web_app` attribute.
        """
    self.serverapp = serverapp
    self.load_config_file()
    self.update_config(self.serverapp.config)
    self.parse_command_line(self.serverapp.extra_args)
    self.serverapp.update_config(self.config)
    self._linked = True