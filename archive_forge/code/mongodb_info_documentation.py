from __future__ import absolute_import, division, print_function
from uuid import UUID
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible.module_utils.six import iteritems
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
Gather parameters information.

        Returns a dictionary with parameters.
        