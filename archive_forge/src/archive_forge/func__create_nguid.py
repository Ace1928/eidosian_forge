from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.common import (
def _create_nguid(serial):
    nguid = 'eui.00' + serial[0:14] + '24a937' + serial[-10:]
    return nguid