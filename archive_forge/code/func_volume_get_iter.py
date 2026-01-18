from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def volume_get_iter(self, vol_name=None):
    """
        Return volume-get-iter query results
        :param vol_name: name of the volume
        :return: NaElement
        """
    volume_info = netapp_utils.zapi.NaElement('volume-get-iter')
    volume_attributes = netapp_utils.zapi.NaElement('volume-attributes')
    volume_id_attributes = netapp_utils.zapi.NaElement('volume-id-attributes')
    volume_id_attributes.add_new_child('name', vol_name)
    volume_id_attributes.add_new_child('vserver', self.parameters['vserver'])
    volume_attributes.add_child_elem(volume_id_attributes)
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(volume_attributes)
    volume_info.add_child_elem(query)
    try:
        result = self.server.invoke_successfully(volume_info, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching volume %s : %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
    return result