import os
import sys
from copy import copy
from jupyter_core.application import JupyterApp, base_aliases, base_flags
from traitlets import Bool, Instance, List, Unicode, default
from jupyterlab.coreconfig import CoreConfig
from jupyterlab.debuglog import DebugLogFileMixin
from .commands import (
from .federated_labextensions import build_labextension, develop_labextension_py, watch_labextension
from .labapp import LabApp
class WatchLabExtensionApp(BaseExtensionApp):
    description = '(developer) Watch labextension'
    development = Bool(True, config=True, help='Build in development mode')
    source_map = Bool(False, config=True, help='Generate source maps')
    core_path = Unicode(os.path.join(HERE, 'staging'), config=True, help='Directory containing core application package.json file')
    aliases = {'core-path': 'WatchLabExtensionApp.core_path', 'development': 'WatchLabExtensionApp.development', 'source-map': 'WatchLabExtensionApp.source_map'}

    def run_task(self):
        self.extra_args = self.extra_args or [os.getcwd()]
        labextensions_path = self.labextensions_path
        watch_labextension(self.extra_args[0], labextensions_path, logger=self.log, development=self.development, source_map=self.source_map, core_path=self.core_path or None)