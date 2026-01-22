from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def addvolumecopy(self):
    self.log('Entering function addvolumecopy')
    cmd = 'addvolumecopy'
    cmdopts = {}
    if self.compressed:
        cmdopts['compressed'] = self.compressed
    if self.grainsize:
        cmdopts['grainsize'] = self.grainsize
    if self.thin and self.rsize:
        cmdopts['thin'] = self.thin
        cmdopts['buffersize'] = self.rsize
    elif self.thin:
        cmdopts['thin'] = self.thin
    elif self.rsize and (not self.thin):
        self.module.fail_json(msg="To configure 'rsize', parameter 'thin' should be passed and the value should be 'true'.")
    if self.deduplicated:
        cmdopts['deduplicated'] = self.deduplicated
    if self.size:
        self.module.fail_json(msg="Parameter 'size' cannot be passed while converting a standard volume to Mirror Volume")
    if self.poolA and (self.poolB == self.discovered_standard_vol_pool and self.poolA != self.discovered_standard_vol_pool):
        cmdopts['pool'] = self.poolA
    elif self.poolB and (self.poolA == self.discovered_standard_vol_pool and self.poolB != self.discovered_standard_vol_pool):
        cmdopts['pool'] = self.poolB
    if self.module.check_mode:
        self.changed = True
        return
    cmdargs = [self.name]
    self.restapi.svc_run_command(cmd, cmdopts, cmdargs)