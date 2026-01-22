from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _delete_server_snapshot(self, server):
    """
        Delete snapshot for the CLC server
        :param server: the CLC server object
        :return: the delete snapshot request object from CLC API
        """
    result = None
    try:
        result = server.DeleteSnapshot()
    except CLCException as ex:
        self.module.fail_json(msg='Failed to delete snapshot for server : {0}. {1}'.format(server.id, ex.response_text))
    return result