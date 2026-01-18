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
def validate_nlfs(self):
    self.parameters['license_codes'] = self.split_nlfs()
    nlf_count = 0
    for license in self.parameters['license_codes']:
        nlf, nlf_dict, is_nlf = self.scan_license_codes_for_nlf(license)
        if is_nlf and (not self.use_rest):
            self.module.fail_json(msg='Error: NLF license format is not supported with ZAPI.')
        self.nlfs.append((nlf, nlf_dict))
        if is_nlf:
            nlf_count += 1
    if nlf_count and nlf_count != len(self.parameters['license_codes']):
        self.module.fail_json(msg='Error: cannot mix legacy licenses and NLF licenses; found %d NLF licenses out of %d license_codes.' % (nlf_count, len(self.parameters['license_codes'])))