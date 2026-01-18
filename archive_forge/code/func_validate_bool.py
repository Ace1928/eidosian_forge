from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
from ansible.module_utils.common import validation
from abc import ABCMeta, abstractmethod
import os.path
import copy
import json
import inspect
import re
def validate_bool(item, param_spec, param_name, invalid_params):
    """
    This function checks that the input `item` is a valid boolean value. If it does
    not represent a valid boolean value, an error message is added to `invalid_params`.

    Args:
        item (bool): The input boolean value to be validated.
        param_spec (dict): The parameter's specification, including validation constraints.
        param_name (str): The name of the parameter being validated.
        invalid_params (list): A list to collect validation error messages.

    Returns:
        bool: The validated boolean value.
    """
    return validation.check_type_bool(item)