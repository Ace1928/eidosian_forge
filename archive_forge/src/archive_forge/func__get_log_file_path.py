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
def _get_log_file_path(conf, binary=None):
    logfile = conf.log_file
    logdir = conf.log_dir
    if logfile and (not logdir):
        return logfile
    if logfile and logdir:
        return os.path.join(logdir, logfile)
    if logdir:
        binary = binary or handlers._get_binary_name()
        return '%s.log' % (os.path.join(logdir, binary),)
    return None