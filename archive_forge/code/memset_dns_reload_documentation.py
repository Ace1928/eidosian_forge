from __future__ import (absolute_import, division, print_function)
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.memset import memset_api_call

    DNS reloads are a single API call and therefore there's not much
    which can go wrong outside of auth errors.
    