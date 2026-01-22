import configparser
import logging
import logging.handlers
import os
import signal
import sys
from oslo_rootwrap import filters
from oslo_rootwrap import subprocess
def _getlogin():
    try:
        return os.getlogin()
    except OSError:
        return os.getenv('USER') or os.getenv('USERNAME') or os.getenv('LOGNAME')