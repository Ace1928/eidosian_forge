import configparser
import logging
import logging.config
import logging.handlers
import os
import platform
import sys
from oslo_config import cfg
from oslo_utils import eventletutils
from oslo_utils import importutils
from oslo_utils import units
from oslo_log._i18n import _
from oslo_log import _options
from oslo_log import formatters
from oslo_log import handlers
def _load_log_config(log_config_append):
    try:
        if not hasattr(_load_log_config, 'old_time'):
            _load_log_config.old_time = 0
        new_time = os.path.getmtime(log_config_append)
        if _load_log_config.old_time != new_time:
            for logger in _iter_loggers():
                logger.setLevel(logging.NOTSET)
                logger.handlers = []
                logger.propagate = 1
            logging.config.fileConfig(log_config_append, disable_existing_loggers=False)
            _load_log_config.old_time = new_time
    except (configparser.Error, KeyError, os.error, RuntimeError) as exc:
        raise LogConfigError(log_config_append, str(exc))