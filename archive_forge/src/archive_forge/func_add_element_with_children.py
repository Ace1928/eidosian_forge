from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def add_element_with_children(self, element_name, param_name, child_name, ldap_client_create):
    ldap_servers_element = netapp_utils.zapi.NaElement(element_name)
    for ldap_server_name in self.parameters[param_name]:
        ldap_servers_element.add_new_child(child_name, ldap_server_name)
    ldap_client_create.add_child_elem(ldap_servers_element)