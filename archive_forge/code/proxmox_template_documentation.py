from __future__ import absolute_import, division, print_function
import os
import time
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.general.plugins.module_utils.proxmox import (proxmox_auth_argument_spec, ProxmoxAnsible)
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion

        Check the task status and wait until the task is completed or the timeout is reached.
        