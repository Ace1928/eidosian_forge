import dataclasses
import json
import os
import sys
from jupyter_core.application import JupyterApp, NoStart, base_aliases, base_flags
from jupyter_server._version import version_info as jpserver_version_info
from jupyter_server.serverapp import flags
from jupyter_server.utils import url_path_join as ujoin
from jupyterlab_server import (
from notebook_shim.shim import NotebookConfigShimMixin
from traitlets import Bool, Instance, Type, Unicode, default
from ._version import __version__
from .commands import (
from .coreconfig import CoreConfig
from .debuglog import DebugLogFileMixin
from .extensions import MANAGERS as EXT_MANAGERS
from .extensions.manager import PluginManager
from .extensions.readonly import ReadOnlyExtensionManager
from .handlers.announcements import (
from .handlers.build_handler import Builder, BuildHandler, build_path
from .handlers.error_handler import ErrorHandler
from .handlers.extension_manager_handler import ExtensionHandler, extensions_handler_path
from .handlers.plugin_manager_handler import PluginHandler, plugins_handler_path
class LabPathApp(JupyterApp):
    version = version
    description = "\n    Print the configured paths for the JupyterLab application\n\n    The application path can be configured using the JUPYTERLAB_DIR\n        environment variable.\n    The user settings path can be configured using the JUPYTERLAB_SETTINGS_DIR\n        environment variable or it will fall back to\n        `/lab/user-settings` in the default Jupyter configuration directory.\n    The workspaces path can be configured using the JUPYTERLAB_WORKSPACES_DIR\n        environment variable or it will fall back to\n        '/lab/workspaces' in the default Jupyter configuration directory.\n    "

    def start(self):
        print('Application directory:   %s' % get_app_dir())
        print('User Settings directory: %s' % get_user_settings_dir())
        print('Workspaces directory: %s' % get_workspaces_dir())