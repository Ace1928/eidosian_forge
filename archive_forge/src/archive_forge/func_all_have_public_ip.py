from __future__ import absolute_import, division, print_function
import os
import re
import time
import uuid
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def all_have_public_ip(ds, ip_v):
    return all((has_public_ip(d.ip_addresses, ip_v) for d in ds))