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
def aggregate_online(self):
    """
        Set state of an offline aggregate to online
        :return: None
        """
    if self.use_rest:
        return self.patch_aggr_rest('make service state online for', {'state': 'online'})
    online_aggr = netapp_utils.zapi.NaElement.create_node_with_children('aggr-online', **{'aggregate': self.parameters['name'], 'force-online': 'true'})
    try:
        self.server.invoke_successfully(online_aggr, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error changing the state of aggregate %s to %s: %s' % (self.parameters['name'], self.parameters['service_state'], to_native(error)), exception=traceback.format_exc())