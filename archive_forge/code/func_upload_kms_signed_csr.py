from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
def upload_kms_signed_csr(self, kms_signed_csr):
    kms_signed_csr_file = open(kms_signed_csr)
    kms_signed_csr_content = kms_signed_csr_file.read()
    kms_signed_csr_file.close()
    try:
        self.crypto_mgr.UpdateKmsSignedCsrClientCert(self.key_provider_id, kms_signed_csr_content)
    except Exception as e:
        self.module.fail_json(msg="Update KMS signed client CSR cert failed with exception: '%s'" % to_native(e))