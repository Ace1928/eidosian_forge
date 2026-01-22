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
def _refresh_root_level(debug):
    """Set the level of the root logger.

    :param debug: If 'debug' is True, the level will be DEBUG.
     Otherwise the level will be INFO.
    """
    log_root = getLogger(None).logger
    if debug:
        log_root.setLevel(logging.DEBUG)
    else:
        log_root.setLevel(logging.INFO)