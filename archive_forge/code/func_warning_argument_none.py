from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
import time
import http
def warning_argument_none(name, requirement):
    module.warn(f'Cannot get {name} in [default dict], reason: Required argument `{requirement}` not available.')