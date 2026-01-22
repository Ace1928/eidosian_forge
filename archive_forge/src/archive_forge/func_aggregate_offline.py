from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def aggregate_offline(self):
    """
        Set state of an online aggregate to offline
        :return: None
        """
    if self.use_rest:
        return self.patch_aggr_rest('make service state offline for', {'state': 'offline'})
    offline_aggr = netapp_utils.zapi.NaElement.create_node_with_children('aggr-offline', **{'aggregate': self.parameters['name'], 'force-offline': 'false', 'unmount-volumes': str(self.parameters.get('unmount_volumes', False))})
    retry = 10
    while retry > 0:
        try:
            self.server.invoke_successfully(offline_aggr, enable_tunneling=True)
            break
        except netapp_utils.zapi.NaApiError as error:
            if 'disk add operation is in progress' in to_native(error):
                retry -= 1
                if retry > 0:
                    continue
            self.module.fail_json(msg='Error changing the state of aggregate %s to %s: %s' % (self.parameters['name'], self.parameters['service_state'], to_native(error)), exception=traceback.format_exc())