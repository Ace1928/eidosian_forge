from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime

    Drives information is only available for the Legend chassis.
    The Legend chassis product_name has // in it so only bother if
    that is the case.
    