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
class InstallLabExtensionApp(BaseExtensionApp):
    description = 'Install labextension(s)\n\n     Usage\n\n        jupyter labextension install [--pin-version-as <alias,...>] <package...>\n\n    This installs JupyterLab extensions similar to yarn add or npm install.\n\n    Pass a list of comma separate names to the --pin-version-as flag\n    to use as aliases for the packages providers. This is useful to\n    install multiple versions of the same extension.\n    These can be uninstalled with the alias you provided\n    to the flag, similar to the "alias" feature of yarn add.\n    '
    aliases = install_aliases
    pin = Unicode('', config=True, help='Pin this version with a certain alias')

    def run_task(self):
        self.deprecation_warning('Installing extensions with the jupyter labextension install command is now deprecated and will be removed in a future major version of JupyterLab.')
        pinned_versions = self.pin.split(',')
        self.extra_args = self.extra_args or [os.getcwd()]
        return any((install_extension(arg, pin=pinned_versions[i] if i < len(pinned_versions) else None, app_options=AppOptions(app_dir=self.app_dir, logger=self.log, core_config=self.core_config, labextensions_path=self.labextensions_path)) for i, arg in enumerate(self.extra_args)))