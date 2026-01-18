from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.basic import AnsibleModule

    Removes a Datacenter.

    This will remove a datacenter.

    module : AnsibleModule object
    profitbricks: authenticated profitbricks object.

    Returns:
        True if the datacenter was deleted, false otherwise
    