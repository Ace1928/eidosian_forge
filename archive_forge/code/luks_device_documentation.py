from __future__ import (absolute_import, division, print_function)
import os
import re
import stat
from ansible.module_utils.basic import AnsibleModule
 If luks is already opened, return its name.
            If 'name' parameter is specified and differs
            from obtained value, fail.
            Return None otherwise
        