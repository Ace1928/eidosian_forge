import configparser
import logging
import warnings
from configparser import ConfigParser
from pathlib import Path
from project import ProjectData
from .commands import run_qmlimportscanner
from . import DEFAULT_APP_ICON
def _find_and_set_project_file(self):
    if self.project_dir:
        files = list(self.project_dir.glob('*.pyproject'))
    else:
        logging.exception('[DEPLOY] Project directory not set in config file')
        raise
    if not files:
        logging.info('[DEPLOY] No .pyproject file found. Project file not set')
    elif len(files) > 1:
        logging.warning('DEPLOY: More that one .pyproject files found. Project file not set')
        raise
    else:
        self.project_data = ProjectData(files[0])
        self.set_value('app', 'project_file', str(files[0].relative_to(self.project_dir)))
        logging.info(f'[DEPLOY] Project file {files[0]} found and set in config file')