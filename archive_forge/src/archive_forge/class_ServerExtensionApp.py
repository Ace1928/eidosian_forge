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
class ServerExtensionApp(BaseExtensionApp):
    """Root level server extension app"""
    name = 'jupyter server extension'
    version = __version__
    description: str = 'Work with Jupyter server extensions'
    examples = _examples
    subcommands: dict[str, t.Any] = {'enable': (EnableServerExtensionApp, 'Enable a server extension'), 'disable': (DisableServerExtensionApp, 'Disable a server extension'), 'list': (ListServerExtensionsApp, 'List server extensions')}

    def start(self) -> None:
        """Perform the App's actions as configured"""
        super().start()
        subcmds = ', '.join(sorted(self.subcommands))
        sys.exit('Please supply at least one subcommand: %s' % subcmds)