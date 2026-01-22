from __future__ import annotations
import logging
import os
import sys
import typing as t
from jupyter_core.application import JupyterApp
from jupyter_core.paths import ENV_CONFIG_PATH, SYSTEM_CONFIG_PATH, jupyter_config_dir
from tornado.log import LogFormatter
from traitlets import Bool
from jupyter_server._version import __version__
from jupyter_server.extension.config import ExtensionConfigManager
from jupyter_server.extension.manager import ExtensionManager, ExtensionPackage
class BaseExtensionApp(JupyterApp):
    """Base extension installer app"""
    _log_formatter_cls = LogFormatter
    flags = _base_flags
    aliases = _base_aliases
    version = __version__
    user = Bool(False, config=True, help='Whether to do a user install')
    sys_prefix = Bool(True, config=True, help='Use the sys.prefix as the prefix')
    python = Bool(False, config=True, help='Install from a Python package')

    def _log_format_default(self) -> str:
        """A default format for messages"""
        return '%(message)s'

    @property
    def config_dir(self) -> str:
        return _get_config_dir(user=self.user, sys_prefix=self.sys_prefix)