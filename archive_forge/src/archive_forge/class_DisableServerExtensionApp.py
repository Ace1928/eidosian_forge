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
class DisableServerExtensionApp(ToggleServerExtensionApp):
    """An App that disables Server Extensions"""
    name = 'jupyter server extension disable'
    description = '\n    Disable a server extension in configuration.\n\n    Usage\n        jupyter server extension disable [--system|--sys-prefix]\n    '
    _toggle_value = False
    _toggle_pre_message = 'disabling'
    _toggle_post_message = 'disabled'