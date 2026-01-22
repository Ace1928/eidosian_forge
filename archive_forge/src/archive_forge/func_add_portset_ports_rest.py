from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def add_portset_ports_rest(self, portset_uuid, ports_to_add):
    """
        Add the list of ports to portset
        :return: None
        """
    api = 'protocols/san/portsets/%s/interfaces' % portset_uuid
    body = {'records': []}
    for port in ports_to_add:
        body['records'].append({self.desired_lifs[port]['lif_type']: {'name': port}})
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error:
        self.module.fail_json(msg=error)