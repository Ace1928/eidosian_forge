from __future__ import absolute_import, division, print_function
import re
import uuid
import time
from ansible.module_utils.basic import AnsibleModule
def _make_default_name():
    return str(uuid.uuid4()).replace('-', '')[:10]