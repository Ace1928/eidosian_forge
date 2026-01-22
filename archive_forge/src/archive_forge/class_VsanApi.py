from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
class VsanApi(PyVmomi):

    def __init__(self, module):
        super(VsanApi, self).__init__(module)
        client_stub = self.si._GetStub()
        ssl_context = client_stub.schemeArgs.get('context')
        apiVersion = vsanapiutils.GetLatestVmodlVersion(module.params['hostname'])
        vcMos = vsanapiutils.GetVsanVcMos(client_stub, context=ssl_context, version=apiVersion)
        self.vsanVumSystem = vcMos['vsan-vum-system']

    def upload_release_catalog(self, content):
        self.vsanVumSystem.VsanVcUploadReleaseDb(db=content)