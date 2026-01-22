from __future__ import absolute_import, division, print_function
import re
import sys
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def add_licenses(self):
    """
        Add licenses
        """
    if self.use_rest:
        return self.add_licenses_rest()
    license_add = netapp_utils.zapi.NaElement('license-v2-add')
    codes = netapp_utils.zapi.NaElement('codes')
    for code in self.parameters['license_codes']:
        codes.add_new_child('license-code-v2', str(code.strip().lower()))
    license_add.add_child_elem(codes)
    try:
        self.server.invoke_successfully(license_add, enable_tunneling=False)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error adding licenses: %s' % to_native(error), exception=traceback.format_exc())