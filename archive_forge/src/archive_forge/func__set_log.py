from __future__ import (absolute_import, division, print_function)
import logging
import os.path
import subprocess
from subprocess import call
import sys
import time
from configparser import ConfigParser
from bcolors import bcolors
def _set_log(self, log_file, log_level):
    logger = logging.getLogger(PREFIX)
    if log_file is not None and log_file != '':
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr = logging.FileHandler(log_file)
        hdlr.setFormatter(formatter)
    else:
        hdlr = logging.StreamHandler(sys.stdout)
    logger.addHandler(hdlr)
    logger.setLevel(log_level)
    return logger