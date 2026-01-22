from __future__ import absolute_import, division, print_function
import re
import uuid
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import xrange
from ansible.module_utils.common.text.converters import to_native
def _create_datacenter(module, profitbricks):
    datacenter = module.params.get('datacenter')
    location = module.params.get('location')
    wait_timeout = module.params.get('wait_timeout')
    i = Datacenter(name=datacenter, location=location)
    try:
        datacenter_response = profitbricks.create_datacenter(datacenter=i)
        _wait_for_completion(profitbricks, datacenter_response, wait_timeout, '_create_datacenter')
        return datacenter_response
    except Exception as e:
        module.fail_json(msg='failed to create the new server(s): %s' % str(e))