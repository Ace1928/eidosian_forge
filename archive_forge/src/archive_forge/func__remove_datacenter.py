from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.basic import AnsibleModule
def _remove_datacenter(module, profitbricks, datacenter):
    try:
        profitbricks.delete_datacenter(datacenter)
    except Exception as e:
        module.fail_json(msg='failed to remove the datacenter: %s' % str(e))