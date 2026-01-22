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
class LabCleanApp(JupyterApp):
    version = version
    description = '\n    Clean the JupyterLab application\n\n    This will clean the app directory by removing the `staging` directories.\n    Optionally, the `extensions`, `settings`, and/or `static` directories,\n    or the entire contents of the app directory, can also be removed.\n    '
    aliases = clean_aliases
    flags = clean_flags
    core_config = Instance(CoreConfig, allow_none=True)
    app_dir = Unicode('', config=True, help='The app directory to clean')
    extensions = Bool(False, config=True, help='Also delete <app-dir>/extensions.\n%s' % ext_warn_msg)
    settings = Bool(False, config=True, help='Also delete <app-dir>/settings')
    static = Bool(False, config=True, help='Also delete <app-dir>/static')
    all = Bool(False, config=True, help='Delete the entire contents of the app directory.\n%s' % ext_warn_msg)

    def start(self):
        app_options = LabCleanAppOptions(logger=self.log, core_config=self.core_config, app_dir=self.app_dir, extensions=self.extensions, settings=self.settings, static=self.static, all=self.all)
        clean(app_options=app_options)