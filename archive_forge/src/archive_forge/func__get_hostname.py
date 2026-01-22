from __future__ import (absolute_import, division, print_function)
import hashlib
import json
import re
import uuid
import os
from collections import namedtuple
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable
from ansible.module_utils.six import iteritems
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMAuth
from ansible.errors import AnsibleParserError, AnsibleError
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils._text import to_native, to_bytes, to_text
from itertools import chain
def _get_hostname(self, host, hostnames=None, strict=False):
    hostname = None
    errors = []
    for preference in hostnames:
        if preference == 'default':
            return host.default_inventory_hostname
        try:
            hostname = self._compose(preference, host.hostvars)
        except Exception as e:
            if strict:
                raise AnsibleError('Could not compose %s as hostnames - %s' % (preference, to_native(e)))
            else:
                errors.append((preference, str(e)))
        if hostname:
            return to_text(hostname)
    raise AnsibleError('Could not template any hostname for host, errors for each preference: %s' % ', '.join(['%s: %s' % (pref, err) for pref, err in errors]))