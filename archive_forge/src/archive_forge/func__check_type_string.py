from __future__ import absolute_import, division, print_function
import os
import re
import logging
import sys
from copy import deepcopy
from ansible.module_utils.basic import env_fallback
def _check_type_string(x):
    """
    :param x:
    :return: True if it is of type string
    """
    if isinstance(x, str):
        return True
    if sys.version_info[0] < 3:
        try:
            return isinstance(x, unicode)
        except NameError:
            return False