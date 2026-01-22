from __future__ import absolute_import, division, print_function
import re
import uuid
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import xrange
from ansible.module_utils.common.text.converters import to_native
def _remove_boot_volume(module, profitbricks, datacenter_id, server_id):
    """
    Remove the boot volume from the server
    """
    try:
        server = profitbricks.get_server(datacenter_id, server_id)
        volume_id = server['properties']['bootVolume']['id']
        volume_response = profitbricks.delete_volume(datacenter_id, volume_id)
    except Exception as e:
        module.fail_json(msg="failed to remove the server's boot volume: %s" % to_native(e), exception=traceback.format_exc())