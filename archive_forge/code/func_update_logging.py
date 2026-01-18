import logging
from warnings import warn
import os
import sys
from .misc import str2bool
def update_logging(self, config):
    self._config = config
    self.disable_file_logging()
    self._logger.setLevel(logging.getLevelName(config.get('logging', 'workflow_level')))
    self._utlogger.setLevel(logging.getLevelName(config.get('logging', 'utils_level')))
    self._iflogger.setLevel(logging.getLevelName(config.get('logging', 'interface_level')))
    if str2bool(config.get('logging', 'log_to_file')):
        self.enable_file_logging()