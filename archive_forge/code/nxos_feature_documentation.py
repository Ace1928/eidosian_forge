from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import ConnectionError
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
Some features may need to be mapped due to inconsistency
    between how they appear from "show feature" output and
    how they are configured