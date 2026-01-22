from __future__ import absolute_import, division, print_function
import hashlib
import json
import os
import operator
import re
import time
import traceback
from contextlib import contextmanager
from collections import defaultdict
from functools import wraps
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, env_fallback
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils import six
def _foreman_spec_helper(spec):
    """Extend an entity spec by adding entries for all flat_names.
    Extract Ansible compatible argument_spec on the way.
    """
    foreman_spec = {}
    argument_spec = {}
    _FILTER_SPEC_KEYS = {'ensure', 'failsafe', 'flat_name', 'foreman_spec', 'invisible', 'optional_scope', 'resolve', 'resource_type', 'scope', 'search_by', 'search_operator', 'thin', 'type'}
    _VALUE_SPEC_KEYS = {'ensure', 'type'}
    _ENTITY_SPEC_KEYS = {'failsafe', 'optional_scope', 'resolve', 'resource_type', 'scope', 'search_by', 'search_operator', 'thin'}
    for key, value in spec.items():
        foreman_value = {k: v for k, v in value.items() if k in _VALUE_SPEC_KEYS}
        argument_value = {k: v for k, v in value.items() if k not in _FILTER_SPEC_KEYS}
        foreman_type = value.get('type')
        ansible_invisible = value.get('invisible', False)
        flat_name = value.get('flat_name')
        if foreman_type == 'entity':
            if not flat_name:
                flat_name = '{0}_id'.format(key)
            foreman_value['resource_type'] = HAS_APYPIE and inflector.pluralize(key)
            foreman_value.update({k: v for k, v in value.items() if k in _ENTITY_SPEC_KEYS})
        elif foreman_type == 'entity_list':
            argument_value['type'] = 'list'
            argument_value['elements'] = value.get('elements', 'str')
            if not flat_name:
                flat_name = '{0}_ids'.format(HAS_APYPIE and inflector.singularize(key))
            foreman_value['resource_type'] = key
            foreman_value.update({k: v for k, v in value.items() if k in _ENTITY_SPEC_KEYS})
        elif foreman_type == 'nested_list':
            argument_value['type'] = 'list'
            argument_value['elements'] = 'dict'
            foreman_value['foreman_spec'], argument_value['options'] = _foreman_spec_helper(value['foreman_spec'])
            foreman_value['ensure'] = value.get('ensure', False)
        elif foreman_type:
            argument_value['type'] = foreman_type
        if flat_name:
            foreman_value['flat_name'] = flat_name
            foreman_spec[flat_name] = {}
            if argument_value.get('type') is not None:
                foreman_spec[flat_name]['type'] = argument_value['type']
        foreman_spec[key] = foreman_value
        if not ansible_invisible:
            argument_spec[key] = argument_value
    return (foreman_spec, argument_spec)