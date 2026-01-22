from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def add_proxy_details(self):
    existed = []
    if self.support == 'remote':
        cmd = 'mksystemsupportcenter'
        cmdargs = []
        for nm, ip, port in zip(self.name, self.sra_ip, self.sra_port):
            if nm != 'None' and ip != 'None' and (port != 'None'):
                if not self.is_proxy_exist(nm):
                    existed.append(True)
                    if not self.is_sra_enabled():
                        cmdopts = {'name': nm, 'ip': ip, 'port': port, 'proxy': 'yes'}
                        self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
                        self.log('Proxy server(%s) details added', nm)
                else:
                    self.log('Skipping, Proxy server(%s) already exist', nm)
            else:
                missing_params = ', '.join([k for k, v in self.filtered_params.items() if 'None' in v])
                self.module.fail_json(msg='support is remote and state is enabled but following parameter missing: {0}'.format(missing_params))
    return existed