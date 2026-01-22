import configparser
import logging
import warnings
from configparser import ConfigParser
from pathlib import Path
from project import ProjectData
from .commands import run_qmlimportscanner
from . import DEFAULT_APP_ICON
def _find_and_set_exe_dir(self):
    if self.project_dir == Path.cwd():
        self.exe_dir = self.project_dir.relative_to(Path.cwd())
    else:
        self.exe_dir = self.project_dir
    self.exe_dir = Path(self.set_or_fetch(config_property_val=self.exe_dir, config_property_key='exec_directory')).absolute()