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
def _prepare_settings(self):
    """Prepare the settings."""
    assert self.serverapp is not None
    webapp = self.serverapp.web_app
    self.settings.update(**webapp.settings)
    self.settings.update({f'{self.name}_static_paths': self.static_paths, f'{self.name}': self})
    self.initialize_settings()
    webapp.settings.update(**self.settings)