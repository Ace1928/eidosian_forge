from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def vserver_peer_create(self):
    """
        Create a vserver peer
        """
    if self.parameters.get('applications') is None:
        self.module.fail_json(msg='applications parameter is missing')
    if self.parameters.get('peer_cluster') is None:
        self.parameters['peer_cluster'] = self.get_peer_cluster_name()
    if self.use_rest:
        return self.vserver_peer_create_rest()
    vserver_peer_create = netapp_utils.zapi.NaElement.create_node_with_children('vserver-peer-create', **{'peer-vserver': self.parameters['peer_vserver'], 'vserver': self.parameters['vserver'], 'peer-cluster': self.parameters['peer_cluster']})
    if 'local_name_for_peer' in self.parameters:
        vserver_peer_create.add_new_child('local-name', self.parameters['local_name_for_peer'])
    applications = netapp_utils.zapi.NaElement('applications')
    for application in self.parameters['applications']:
        applications.add_new_child('vserver-peer-application', application)
    vserver_peer_create.add_child_elem(applications)
    try:
        self.server.invoke_successfully(vserver_peer_create, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error creating vserver peer %s: %s' % (self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())