from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def vserver_peer_delete_rest(self, current):
    """
        Delete a vserver peer using rest.
        """
    dummy, error = rest_generic.delete_async(self.rest_api, 'svm/peers', current['local_peer_vserver_uuid'])
    self.check_and_report_rest_error(error, 'deleting', self.parameters['vserver'])