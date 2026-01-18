from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
from ansible.module_utils.common import validation
from abc import ABCMeta, abstractmethod
import os.path
import copy
import json
import inspect
import re
def validate_list(item, param_spec, param_name, invalid_params):
    """
    This function checks if the input `item` is a valid list based on the specified `param_spec`.
    It also verifies that the elements of the list match the expected data type specified in the
    `param_spec`. If any validation errors occur, they are appended to the `invalid_params` list.

    Args:
        item (list): The input list to be validated.
        param_spec (dict): The parameter's specification, including validation constraints.
        param_name (str): The name of the parameter being validated.
        invalid_params (list): A list to collect validation error messages.

    Returns:
        list: The validated list, potentially normalized based on the specification.
    """
    try:
        if param_spec.get('type') == type(item).__name__:
            keys_list = []
            for dict_key in param_spec:
                keys_list.append(dict_key)
            if len(keys_list) == 1:
                return validation.check_type_list(item)
            temp_dict = {keys_list[1]: param_spec[keys_list[1]]}
            try:
                if param_spec['elements']:
                    get_spec_type = param_spec['type']
                    get_spec_element = param_spec['elements']
                    if type(item).__name__ == get_spec_type:
                        for element in item:
                            if type(element).__name__ != get_spec_element:
                                invalid_params.append('{0} is not of the same datatype as expected which is {1}'.format(element, get_spec_element))
                    else:
                        invalid_params.append('{0} is not of the same datatype as expected which is {1}'.format(item, get_spec_type))
            except Exception as e:
                item, list_invalid_params = validate_list_of_dicts(item, temp_dict)
                invalid_params.extend(list_invalid_params)
        else:
            invalid_params.append('{0} : is not a valid list'.format(item))
    except Exception as e:
        invalid_params.append('{0} : comes into the exception'.format(e))
    return item