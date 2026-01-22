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
def _revert_entity(self, resource, current_entity, params):
    """
        Revert a given entity

        :param resource: Plural name of the api resource to manipulate
        :type resource: str
        :param current_entity: Current properties of the entity
        :type current_entity: dict
        :param params: Lookup parameters (i.e. parent_id for nested entities)
        :type params: dict, optional

        :return: The new current state of the entity
        :rtype: dict
        """
    payload = {'id': current_entity['id']}
    if params:
        payload.update(params)
    return self.resource_action(resource, 'revert', payload)