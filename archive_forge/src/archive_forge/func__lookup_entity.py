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
def _lookup_entity(self, identifier, entity_spec, params=None):
    if identifier is NoEntity:
        return NoEntity
    resource_type = entity_spec['resource_type']
    failsafe = entity_spec.get('failsafe', False)
    thin = entity_spec.get('thin', True)
    if params:
        params = params.copy()
    else:
        params = {}
    try:
        for scope in entity_spec.get('scope', []):
            params.update(self.scope_for(scope, resource_type))
        for optional_scope in entity_spec.get('optional_scope', []):
            if optional_scope in self.foreman_params:
                params.update(self.scope_for(optional_scope, resource_type))
    except TypeError:
        if failsafe:
            if entity_spec.get('type') == 'entity':
                result = None
            else:
                result = [None for value in identifier]
        else:
            self.fail_json(msg='Failed to lookup scope {0} while searching for {1}.'.format(entity_spec['scope'], resource_type))
    else:
        if resource_type == 'operatingsystems':
            if entity_spec.get('type') == 'entity':
                result = self.find_operatingsystem(identifier, params=params, failsafe=failsafe, thin=thin)
            else:
                result = [self.find_operatingsystem(value, params=params, failsafe=failsafe, thin=thin) for value in identifier]
        elif resource_type == 'puppetclasses':
            if entity_spec.get('type') == 'entity':
                result = self.find_puppetclass(identifier, params=params, failsafe=failsafe, thin=thin)
            else:
                result = [self.find_puppetclass(value, params=params, failsafe=failsafe, thin=thin) for value in identifier]
        elif entity_spec.get('type') == 'entity':
            result = self.find_resource_by(resource=resource_type, value=identifier, search_field=entity_spec.get('search_by', ENTITY_KEYS.get(resource_type, 'name')), search_operator=entity_spec.get('search_operator', '='), failsafe=failsafe, thin=thin, params=params)
        else:
            result = [self.find_resource_by(resource=resource_type, value=value, search_field=entity_spec.get('search_by', ENTITY_KEYS.get(resource_type, 'name')), search_operator=entity_spec.get('search_operator', '='), failsafe=failsafe, thin=thin, params=params) for value in identifier]
    return result