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
def _prepare_config(self):
    """Builds a Config object from the extension's traits and passes
        the object to the webapp's settings as `<name>_config`.
        """
    traits = self.class_own_traits().keys()
    self.extension_config = Config({t: getattr(self, t) for t in traits})
    self.settings[f'{self.name}_config'] = self.extension_config