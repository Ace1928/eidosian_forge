from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (

                We need to wait for Active snapshot ID, to be removed as it's current
                stateless snapshot. Then we need to wait for staless snapshot ID to
                be read, for use, because it will become active snapshot.
                