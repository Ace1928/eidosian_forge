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
class ListLabExtensionsApp(BaseExtensionApp):
    description = 'List the installed labextensions'

    def run_task(self):
        list_extensions(app_options=AppOptions(app_dir=self.app_dir, logger=self.log, core_config=self.core_config, labextensions_path=self.labextensions_path))