from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
from ansible.module_utils.common import validation
from abc import ABCMeta, abstractmethod
import os.path
import copy
import json
import inspect
import re
def validate_int(item, param_spec, param_name, invalid_params):
    """
    This function checks that the input `item` is a valid integer and conforms to
    the constraints specified in `param_spec`. If the integer is not valid or does
    not meet the constraints, an error message is added to `invalid_params`.

    Args:
        item (int): The input integer to be validated.
        param_spec (dict): The parameter's specification, including validation constraints.
        param_name (str): The name of the parameter being validated.
        invalid_params (list): A list to collect validation error messages.

    Returns:
        int: The validated integer.

    Example `param_spec`:
        {
            "type": "int",
            "range_min": 1,     # Optional: minimum allowed value
            "range_max": 100    # Optional: maximum allowed value
        }
    """
    item = validation.check_type_int(item)
    min_value = 1
    if param_spec.get('range_min') is not None:
        min_value = param_spec.get('range_min')
    if param_spec.get('range_max'):
        if min_value <= item <= param_spec.get('range_max'):
            return item
        else:
            invalid_params.append('{0}:{1} : The item exceeds the allowed range of max {2}'.format(param_name, item, param_spec.get('range_max')))
    return item