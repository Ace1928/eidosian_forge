from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def add_portset_ports(self, port):
    """
        Add the list of ports to portset
        :return: None
        """
    options = {'portset-name': self.parameters['resource_name'], 'portset-port-name': port.strip()}
    portset_modify = netapp_utils.zapi.NaElement.create_node_with_children('portset-add', **options)
    try:
        self.server.invoke_successfully(portset_modify, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error adding port in portset %s: %s' % (self.parameters['resource_name'], to_native(error)), exception=traceback.format_exc())