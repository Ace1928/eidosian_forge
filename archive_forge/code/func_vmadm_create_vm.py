from __future__ import absolute_import, division, print_function
import json
import os
import re
import tempfile
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def vmadm_create_vm(module, payload_file):
    cmd = [module.vmadm, 'create', '-f', payload_file]
    return module.run_command(cmd)