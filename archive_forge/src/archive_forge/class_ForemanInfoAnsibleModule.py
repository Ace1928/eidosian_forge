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
class ForemanInfoAnsibleModule(ForemanStatelessEntityAnsibleModule):
    """
    Base class for Foreman info modules that fetch information about entities
    """

    def __init__(self, **kwargs):
        self._resources = []
        foreman_spec = dict(name=dict(), search=dict(), organization=dict(type='entity'), location=dict(type='entity'))
        foreman_spec.update(kwargs.pop('foreman_spec', {}))
        mutually_exclusive = kwargs.pop('mutually_exclusive', [])
        if not foreman_spec['name'].get('invisible', False):
            mutually_exclusive.extend([['name', 'search']])
        super(ForemanInfoAnsibleModule, self).__init__(foreman_spec=foreman_spec, mutually_exclusive=mutually_exclusive, **kwargs)

    def run(self, **kwargs):
        """
        lookup entities
        """
        self.auto_lookup_entities()
        resource = self.foreman_spec['entity']['resource_type']
        if 'name' in self.foreman_params:
            self._info_result = {self.entity_name: self.lookup_entity('entity')}
        else:
            _flat_entity = _flatten_entity(self.foreman_params, self.foreman_spec)
            self._info_result = {resource: self.list_resource(resource, self.foreman_params.get('search'), _flat_entity)}

    def exit_json(self, **kwargs):
        kwargs.update(self._info_result)
        super(ForemanInfoAnsibleModule, self).exit_json(**kwargs)