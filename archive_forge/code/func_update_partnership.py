from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def update_partnership(self, location, id, modify_data):
    if self.module.check_mode:
        self.changed = True
        return
    cmd = 'chpartnership'
    cmd_args = [id]
    rest_object = None
    if location == 'local':
        rest_object = self.restapi_local
    if location == 'remote':
        rest_object = self.restapi_remote
    if 'compressed' in modify_data or 'clusterip' in modify_data:
        cmd_opts = {}
        if 'compressed' in modify_data:
            cmd_opts['compressed'] = modify_data['compressed']
        if 'clusterip' in modify_data and location == 'local':
            cmd_opts['clusterip'] = modify_data['clusterip']
        if cmd_opts:
            self.stop_partnership(rest_object, id)
            rest_object.svc_run_command(cmd, cmd_opts, cmd_args)
            self.start_partnership(rest_object, id)
            self.changed = True
    if 'linkbandwidthmbits' in modify_data or 'backgroundcopyrate' in modify_data:
        cmd_opts = {}
        if 'linkbandwidthmbits' in modify_data:
            cmd_opts['linkbandwidthmbits'] = modify_data['linkbandwidthmbits']
        if 'backgroundcopyrate' in modify_data:
            cmd_opts['backgroundcopyrate'] = modify_data['backgroundcopyrate']
        if cmd_opts:
            rest_object.svc_run_command(cmd, cmd_opts, cmd_args)
            self.changed = True