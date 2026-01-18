from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def validate_rest_path(self):
    """
        REST does not support command or command directory path in ONTAP < 9.11.1 versions.
        """
    invalid_uri = []
    for privilege in self.parameters.get('privileges', []):
        if '/' not in privilege['path']:
            invalid_uri.append(privilege['path'])
    if invalid_uri:
        self.module.fail_json(msg='Error: Invalid URI %s, please set valid REST API path' % invalid_uri)