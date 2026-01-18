from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (

        We need to maintain this type strings, for the __compare_options method,
        for easier comparision.
        