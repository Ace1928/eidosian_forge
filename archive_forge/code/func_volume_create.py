from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def volume_create(self):
    self.log('Entering function volume_create')
    if not self.size:
        self.module.fail_json(msg='You must pass in size to the module.')
    if not self.type:
        self.module.fail_json(msg='You must pass type to the module.')
    self.log("creating Volume '%s'", self.name)
    if self.module.check_mode:
        self.changed = True
        return
    cmd = 'mkvolume'
    cmdopts = {}
    if self.poolA and self.poolB:
        cmdopts['pool'] = self.poolA + ':' + self.poolB
    if self.size:
        cmdopts['size'] = self.size
        cmdopts['unit'] = 'mb'
    if self.grainsize:
        cmdopts['grainsize'] = self.grainsize
    if self.thin and self.rsize:
        cmdopts['thin'] = self.thin
        cmdopts['buffersize'] = self.rsize
    elif self.thin:
        cmdopts['thin'] = self.thin
    elif self.rsize and (not self.thin):
        self.module.fail_json(msg="To configure 'rsize', parameter 'thin' should be passed and the value should be 'true'.")
    if self.compressed:
        cmdopts['compressed'] = self.compressed
    if self.thin:
        cmdopts['thin'] = self.thin
    if self.deduplicated:
        cmdopts['deduplicated'] = self.deduplicated
    cmdopts['name'] = self.name
    self.log('creating volume command %s opts %s', cmd, cmdopts)
    result = self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
    self.log('create volume result %s', result)
    if 'message' in result:
        self.changed = True
        self.log('create volume result message %s', result['message'])
    else:
        self.module.fail_json(msg='Failed to create volume [%s]' % self.name)