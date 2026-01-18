from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
from ansible.module_utils.common import validation
from abc import ABCMeta, abstractmethod
import os.path
import copy
import json
import inspect
import re
def validate_dnac_log_file_path(self):
    """
        Validates the specified log file path, ensuring it is either absolute or relative,
        the directory exists, and has a .log extension.
        """
    dnac_log_file_path = os.path.abspath(self.dnac_log_file_path)
    log_directory = os.path.dirname(dnac_log_file_path)
    if not os.path.exists(log_directory):
        raise FileNotFoundError("The directory for log file '{0}' does not exist.".format(dnac_log_file_path))