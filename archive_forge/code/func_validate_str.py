from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
from ansible.module_utils.common import validation
from abc import ABCMeta, abstractmethod
import os.path
import copy
import json
import inspect
import re
def validate_str(item, param_spec, param_name, invalid_params):
    """
    This function checks that the input `item` is a valid string and confirms to
    the constraints specified in `param_spec`. If the string is not valid or does
    not meet the constraints, an error message is added to `invalid_params`.

    Args:
        item (str): The input string to be validated.
        param_spec (dict): The parameter's specification, including validation constraints.
        param_name (str): The name of the parameter being validated.
        invalid_params (list): A list to collect validation error messages.

    Returns:
        str: The validated and possibly normalized string.

    Example `param_spec`:
        {
            "type": "str",
            "length_max": 255  # Optional: maximum allowed length
        }
    """
    item = validation.check_type_str(item)
    if param_spec.get('length_max'):
        if 1 <= len(item) <= param_spec.get('length_max'):
            return item
        else:
            invalid_params.append('{0}:{1} : The string exceeds the allowed range of max {2} char'.format(param_name, item, param_spec.get('length_max')))
    return item