from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
class IBMSVSSLCertificate:

    def __init__(self):
        argument_spec = svc_argument_spec()
        argument_spec.update(dict(certificate_type=dict(type='str', choices=['system'], default='system')))
        self.module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
        self.certificate_type = self.module.params['certificate_type']
        self.log_path = self.module.params['log_path']
        log = get_logger(self.__class__.__name__, self.log_path)
        self.log = log.info
        self.changed = False
        self.msg = ''
        self.restapi = IBMSVCRestApi(module=self.module, clustername=self.module.params['clustername'], domain=self.module.params['domain'], username=self.module.params['username'], password=self.module.params['password'], validate_certs=self.module.params['validate_certs'], log_path=self.log_path, token=self.module.params['token'])

    def export_cert(self):
        if self.module.check_mode:
            self.changed = True
            return
        self.restapi.svc_run_command('chsystemcert', cmdopts=None, cmdargs=['-export'])
        self.log('Certificate exported')
        self.changed = True

    def apply(self):
        if self.certificate_type == 'system':
            self.export_cert()
            self.msg = 'Certificate exported.'
        if self.module.check_mode:
            self.msg = 'skipping changes due to check mode.'
        self.module.exit_json(changed=self.changed, msg=self.msg)