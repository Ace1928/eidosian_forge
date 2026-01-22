from __future__ import absolute_import, division, print_function
import re
import uuid
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import xrange
from ansible.module_utils.common.text.converters import to_native
def _startstop_machine(module, profitbricks, datacenter_id, server_id):
    state = module.params.get('state')
    try:
        if state == 'running':
            profitbricks.start_server(datacenter_id, server_id)
        else:
            profitbricks.stop_server(datacenter_id, server_id)
        return True
    except Exception as e:
        module.fail_json(msg='failed to start or stop the virtual machine %s at %s: %s' % (server_id, datacenter_id, str(e)))