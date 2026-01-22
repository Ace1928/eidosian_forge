from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _create_server_snapshot(self, server, expiration_days):
    """
        Create the snapshot for the CLC server
        :param server: the CLC server object
        :param expiration_days: The number of days to keep the snapshot
        :return: the create request object from CLC API Call
        """
    result = None
    try:
        result = server.CreateSnapshot(delete_existing=True, expiration_days=expiration_days)
    except CLCException as ex:
        self.module.fail_json(msg='Failed to create snapshot for server : {0}. {1}'.format(server.id, ex.response_text))
    return result