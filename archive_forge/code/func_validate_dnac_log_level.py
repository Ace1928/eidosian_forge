from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
from ansible.module_utils.common import validation
from abc import ABCMeta, abstractmethod
import os.path
import copy
import json
import inspect
import re
def validate_dnac_log_level(self):
    """Validates if the logging level is string and of expected value"""
    if self.dnac_log_level not in ('INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'):
        raise ValueError("Invalid log level: 'dnac_log_level:{0}'".format(self.dnac_log_level))