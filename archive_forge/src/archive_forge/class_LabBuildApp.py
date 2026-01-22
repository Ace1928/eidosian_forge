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
class LabBuildApp(JupyterApp, DebugLogFileMixin):
    version = version
    description = '\n    Build the JupyterLab application\n\n    The application is built in the JupyterLab app directory in `/staging`.\n    When the build is complete it is put in the JupyterLab app `/static`\n    directory, where it is used to serve the application.\n    '
    aliases = build_aliases
    flags = build_flags
    core_config = Instance(CoreConfig, allow_none=True)
    app_dir = Unicode('', config=True, help='The app directory to build in')
    name = Unicode('JupyterLab', config=True, help='The name of the built application')
    version = Unicode('', config=True, help='The version of the built application')
    dev_build = Bool(None, allow_none=True, config=True, help='Whether to build in dev mode. Defaults to True (dev mode) if there are any locally linked extensions, else defaults to False (production mode).')
    minimize = Bool(True, config=True, help='Whether to minimize a production build (defaults to True).')
    pre_clean = Bool(False, config=True, help='Whether to clean before building (defaults to False)')
    splice_source = Bool(False, config=True, help='Splice source packages into app directory.')

    def start(self):
        app_dir = self.app_dir or get_app_dir()
        app_options = AppOptions(app_dir=app_dir, logger=self.log, core_config=self.core_config, splice_source=self.splice_source)
        self.log.info('JupyterLab %s', version)
        with self.debug_logging():
            if self.pre_clean:
                self.log.info('Cleaning %s' % app_dir)
                clean(app_options=app_options)
            self.log.info('Building in %s', app_dir)
            try:
                production = None if self.dev_build is None else not self.dev_build
                build(name=self.name, version=self.version, app_options=app_options, production=production, minimize=self.minimize)
            except Exception as e:
                self.log.error(build_failure_msg)
                raise e