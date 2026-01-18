from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
from ansible.module_utils.common import validation
from abc import ABCMeta, abstractmethod
import os.path
import copy
import json
import inspect
import re
def validate_list_of_dicts(param_list, spec, module=None):
    """Validate/Normalize playbook params. Will raise when invalid parameters found.
    param_list: a playbook parameter list of dicts
    spec: an argument spec dict
          e.g. spec = dict(ip=dict(required=True, type='bool'),
                           foo=dict(type='str', default='bar'))
    return: list of normalized input data
    """
    v = validation
    normalized = []
    invalid_params = []
    for list_entry in param_list:
        valid_params_dict = {}
        if not spec:
            invalid_params.append('No more spec to validate, but parameters remain')
            break
        for param in spec:
            item = list_entry.get(param)
            if item is None:
                if spec[param].get('required'):
                    invalid_params.append('{0} : Required parameter not found'.format(param))
                else:
                    item = spec[param].get('default')
                    valid_params_dict[param] = item
                    continue
            data_type = spec[param].get('type')
            switch = {'str': validate_str, 'int': validate_int, 'bool': validate_bool, 'list': validate_list, 'dict': validate_dict}
            validator = switch.get(data_type)
            if validator:
                item = validator(item, spec[param], param, invalid_params)
            else:
                invalid_params.append('{0}:{1} : Unsupported data type {2}.'.format(param, item, data_type))
            choice = spec[param].get('choices')
            if choice:
                if item not in choice:
                    invalid_params.append('{0} : Invalid choice provided'.format(item))
            no_log = spec[param].get('no_log')
            if no_log:
                if module is not None:
                    module.no_log_values.add(item)
                else:
                    msg = "\n\n'{0}' is a no_log parameter".format(param)
                    msg += '\nAnsible module object must be passed to this '
                    msg += '\nfunction to ensure it is not logged\n\n'
                    raise Exception(msg)
            valid_params_dict[param] = item
        normalized.append(valid_params_dict)
    return (normalized, invalid_params)