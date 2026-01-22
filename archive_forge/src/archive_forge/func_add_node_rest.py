from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def add_node_rest(self):
    """
        Add a node to an existing cluster
        """
    body = self.create_node_body()
    dummy, error = rest_generic.post_async(self.rest_api, 'cluster/nodes', body, job_timeout=120)
    if error:
        self.module.fail_json(msg='Error adding node with ip %s: %s' % (self.parameters.get('cluster_ip_address'), to_native(error)), exception=traceback.format_exc())