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
class TaxonomyMixin(object):
    """
    Taxonomy Mixin to extend a :class:`ForemanAnsibleModule` (or any subclass) to work with taxonomic entities.

    This adds optional ``organizations`` and ``locations`` parameters to the module.
    """

    def __init__(self, **kwargs):
        foreman_spec = dict(organizations=dict(type='entity_list'), locations=dict(type='entity_list'))
        foreman_spec.update(kwargs.pop('foreman_spec', {}))
        super(TaxonomyMixin, self).__init__(foreman_spec=foreman_spec, **kwargs)