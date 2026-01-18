from __future__ import absolute_import, division, print_function
import base64
import gzip
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
def upload_hcl_db(self, content):
    compressed = gzip.compress(content)
    payload_b64 = base64.b64encode(compressed).decode('ascii')
    self.vsanClusterHealthSystem.VsanVcUploadHclDb(db=payload_b64)