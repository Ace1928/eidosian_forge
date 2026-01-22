from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
class IBMSVCIPPartnership(object):

    def __init__(self):
        argument_spec = svc_argument_spec()
        argument_spec.update(dict(state=dict(type='str', required=True, choices=['present', 'absent']), type=dict(type='str', required=False, choices=['ipv4', 'ipv6']), remote_clusterip=dict(type='str', required=False), remote_cluster_id=dict(type='str', required=False), compressed=dict(type='str', required=False, choices=['yes', 'no']), linkbandwidthmbits=dict(type='int', required=False), backgroundcopyrate=dict(type='int', required=False), link1=dict(type='str', required=False), link2=dict(type='str', required=False), remote_clustername=dict(type='str', required=True), remote_domain=dict(type='str', default=None), remote_username=dict(type='str'), remote_password=dict(type='str', no_log=True), remote_token=dict(type='str', no_log=True), remote_validate_certs=dict(type='bool', default=False), remote_link1=dict(type='str', required=False), remote_link2=dict(type='str', required=False)))
        self.module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
        log_path = self.module.params['log_path']
        log = get_logger(self.__class__.__name__, log_path)
        self.log = log.info
        self.state = self.module.params['state']
        self.remote_clustername = self.module.params['remote_clustername']
        self.remote_username = self.module.params.get('remote_username', '')
        self.remote_password = self.module.params.get('remote_password', '')
        self.remote_clusterip = self.module.params.get('remote_clusterip', '')
        self.remote_cluster_id = self.module.params.get('remote_cluster_id', '')
        self.type = self.module.params.get('type', '')
        self.compressed = self.module.params.get('compressed', '')
        self.linkbandwidthmbits = self.module.params.get('linkbandwidthmbits', '')
        self.backgroundcopyrate = self.module.params.get('backgroundcopyrate', '')
        self.link1 = self.module.params.get('link1', '')
        self.link2 = self.module.params.get('link2', '')
        self.remote_domain = self.module.params.get('remote_domain', '')
        self.remote_token = self.module.params.get('remote_token', '')
        self.remote_validate_certs = self.module.params.get('remote_validate_certs', '')
        self.remote_link1 = self.module.params.get('remote_link1', '')
        self.remote_link2 = self.module.params.get('remote_link2', '')
        self.changed = False
        self.restapi_local = IBMSVCRestApi(module=self.module, clustername=self.module.params['clustername'], domain=self.module.params['domain'], username=self.module.params['username'], password=self.module.params['password'], validate_certs=self.module.params['validate_certs'], log_path=log_path, token=self.module.params['token'])
        self.restapi_remote = IBMSVCRestApi(module=self.module, clustername=self.module.params['remote_clustername'], domain=self.module.params['remote_domain'], username=self.module.params['remote_username'], password=self.module.params['remote_password'], validate_certs=self.module.params['remote_validate_certs'], log_path=log_path, token=self.module.params['remote_token'])

    def basic_checks(self):
        if not self.state:
            self.module.fail_json(msg='Missing mandatory parameter: state')

    def create_parameter_validation(self):
        if self.state == 'present':
            if not self.remote_clusterip:
                self.module.fail_json(msg='Missing required parameter during creation: remote_clusterip')
            if not (self.link1 or self.link2):
                self.module.fail_json(msg='At least one is required during creation: link1 or link2')
            if not (self.remote_link1 or self.remote_link2):
                self.module.fail_json(msg='At least one is required during creation: remote_link1 or remote_link2')

    def delete_parameter_validation(self):
        if self.state == 'absent':
            if not self.remote_cluster_id:
                self.module.fail_json(msg='Missing required parameter during deletion: remote_cluster_id')
            unsupported = []
            check_list = {'remote_clusterip': self.remote_clusterip, 'type': self.type, 'linkbandwidthmbits': self.linkbandwidthmbits, 'backgroundcopyrate': self.backgroundcopyrate, 'compressed': self.compressed, 'link1': self.link1, 'link2': self.link2, 'remote_link1': self.remote_link1, 'remote_link2': self.remote_link2}
            self.log('%s', check_list)
            for key, value in check_list.items():
                if value:
                    unsupported.append(key)
            if unsupported:
                self.module.fail_json(msg='Unsupported parameter during deletion: {0}'.format(unsupported))

    def update_parameter_validation(self):
        if self.state == 'present' and (not self.remote_cluster_id):
            self.module.fail_json(msg='Missing required parameter during updation: remote_cluster_id')

    def get_ip(self, rest_obj):
        system_data = rest_obj.svc_obj_info('lssystem', {}, None)
        if system_data and 'console_IP' in system_data and (':' in system_data['console_IP']):
            return system_data['console_IP'].split(':')[0]
        else:
            self.module.fail_json(msg='Failed to fetch the IP address of local system')

    def get_all_partnership(self, rest_obj):
        return rest_obj.svc_obj_info(cmd='lspartnership', cmdopts=None, cmdargs=[])

    def filter_partnership(self, data, ip):
        return list(filter(lambda item: item['cluster_ip'] == ip, data))

    def get_local_partnership(self, data):
        return list(filter(lambda item: item['location'] == 'local', data))

    def get_partnership_detail(self, rest_obj, id):
        return rest_obj.svc_obj_info(cmd='lspartnership', cmdopts=None, cmdargs=[id])

    def gather_all_validation_data(self, rest_local, rest_remote):
        local_data = {}
        remote_data = {}
        local_ip = self.get_ip(rest_local)
        local_id = None
        if self.remote_cluster_id:
            local_data = self.get_partnership_detail(rest_local, self.remote_cluster_id)
            all_local_partnership = self.get_all_partnership(rest_local)
            if all_local_partnership:
                local_partnership_data = self.get_local_partnership(all_local_partnership)
                if local_partnership_data:
                    local_id = local_partnership_data[0]['id']
                    remote_data = self.get_partnership_detail(rest_remote, local_id)
        else:
            all_local_partnership = self.get_all_partnership(rest_local)
            if all_local_partnership:
                if self.remote_clusterip:
                    local_filter = self.filter_partnership(all_local_partnership, self.remote_clusterip)
                    if local_filter:
                        local_data = self.get_partnership_detail(rest_local, local_filter[0]['id'])
            all_remote_partnership = self.get_all_partnership(rest_remote)
            if all_remote_partnership:
                remote_filter = self.filter_partnership(all_remote_partnership, local_ip)
                if remote_filter:
                    remote_data = self.get_partnership_detail(rest_remote, remote_filter[0]['id'])
        return (local_ip, local_id, local_data, remote_data)

    def create_partnership(self, location, cluster_ip):
        if self.module.check_mode:
            self.changed = True
            return
        rest_api = None
        cmd = 'mkippartnership'
        cmd_opts = {'clusterip': cluster_ip}
        if self.type:
            cmd_opts['type'] = self.type
        if self.compressed:
            cmd_opts['compressed'] = self.compressed
        if self.linkbandwidthmbits:
            cmd_opts['linkbandwidthmbits'] = self.linkbandwidthmbits
        if self.backgroundcopyrate:
            cmd_opts['backgroundcopyrate'] = self.backgroundcopyrate
        if location == 'local':
            rest_api = self.restapi_local
            if self.link1:
                cmd_opts['link1'] = self.link1
            if self.link2:
                cmd_opts['link2'] = self.link2
        if location == 'remote':
            rest_api = self.restapi_remote
            if self.remote_link1:
                cmd_opts['link1'] = self.remote_link1
            if self.remote_link2:
                cmd_opts['link2'] = self.remote_link2
        result = rest_api.svc_run_command(cmd, cmd_opts, cmdargs=None)
        self.log("Create result '%s'.", result)
        if result == '':
            self.changed = True
            self.log('Created IP partnership for %s system.', location)
        else:
            self.module.fail_json(msg='Failed to create IP partnership for cluster ip {0}'.format(cluster_ip))

    def remove_partnership(self, location, id):
        if self.module.check_mode:
            self.changed = True
            return
        rest_api = None
        cmd = 'rmpartnership'
        if location == 'local':
            rest_api = self.restapi_local
        if location == 'remote':
            rest_api = self.restapi_remote
        rest_api.svc_run_command(cmd, {}, [id])
        self.log('Deleted partnership with name %s.', id)
        self.changed = True

    def probe_partnership(self, local_data, remote_data):
        modify_local, modify_remote = ({}, {})
        unsupported = []
        if self.link1:
            if local_data and local_data['link1'] != self.link1:
                unsupported.append('link1')
        if self.link2:
            if local_data and local_data['link2'] != self.link2:
                unsupported.append('link2')
        if self.remote_link1:
            if remote_data and remote_data['link1'] != self.remote_link1:
                unsupported.append('remote_link1')
        if self.remote_link2:
            if remote_data and remote_data['link2'] != self.remote_link2:
                unsupported.append('remote_link2')
        if self.type:
            if local_data and local_data['type'] != self.type or (remote_data and remote_data['type'] != self.type):
                unsupported.append('type')
        if unsupported:
            self.module.fail_json(msg='parameters {0} cannot be updated'.format(unsupported))
        if self.compressed:
            if local_data and local_data['compressed'] != self.compressed:
                modify_local['compressed'] = self.compressed
            if remote_data and remote_data['compressed'] != self.compressed:
                modify_remote['compressed'] = self.compressed
        if self.linkbandwidthmbits:
            if local_data and int(local_data['link_bandwidth_mbits']) != self.linkbandwidthmbits:
                modify_local['linkbandwidthmbits'] = self.linkbandwidthmbits
            if remote_data and int(remote_data['link_bandwidth_mbits']) != self.linkbandwidthmbits:
                modify_remote['linkbandwidthmbits'] = self.linkbandwidthmbits
        if self.backgroundcopyrate:
            if local_data and int(local_data['background_copy_rate']) != self.backgroundcopyrate:
                modify_local['backgroundcopyrate'] = self.backgroundcopyrate
            if remote_data and int(remote_data['background_copy_rate']) != self.backgroundcopyrate:
                modify_remote['backgroundcopyrate'] = self.backgroundcopyrate
        if self.remote_clusterip:
            if local_data and self.remote_clusterip != local_data['cluster_ip']:
                modify_local['clusterip'] = self.remote_clusterip
        return (modify_local, modify_remote)

    def start_partnership(self, rest_object, id):
        cmd = 'chpartnership'
        cmd_opts = {'start': True}
        cmd_args = [id]
        rest_object.svc_run_command(cmd, cmd_opts, cmd_args)
        self.log('Started the partnership %s.', id)

    def stop_partnership(self, rest_object, id):
        cmd = 'chpartnership'
        cmd_opts = {'stop': True}
        cmd_args = [id]
        rest_object.svc_run_command(cmd, cmd_opts, cmd_args)
        self.log('Stopped partnership %s.', id)

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

    def apply(self):
        msg = ''
        self.basic_checks()
        local_ip, local_id, local_data, remote_data = self.gather_all_validation_data(self.restapi_local, self.restapi_remote)
        if self.state == 'present':
            if local_data and remote_data:
                modify_local, modify_remote = self.probe_partnership(local_data, remote_data)
                if modify_local or modify_remote:
                    self.update_parameter_validation()
                    if modify_local:
                        self.update_partnership('local', self.remote_cluster_id, modify_local)
                        msg += 'IP partnership updated on local system.'
                    else:
                        msg += 'IP partnership already exists on local system.'
                    if modify_remote:
                        self.update_partnership('remote', local_id, modify_remote)
                        msg += ' IP partnership updated on remote system.'
                    else:
                        msg += ' IP partnership already exists on remote system.'
                else:
                    msg += 'IP partnership already exists on both local and remote system.'
            elif local_data and (not remote_data):
                response = self.probe_partnership(local_data, remote_data)
                modify_local = response[0]
                self.create_parameter_validation()
                self.create_partnership('remote', local_ip)
                msg += 'IP partnership created on remote system.'
                if modify_local:
                    self.update_parameter_validation()
                    self.update_partnership('local', self.remote_cluster_id, modify_local)
                    msg += ' IP partnership updated on {0} system.'.format(['local'])
                else:
                    msg += ' IP Partnership already exists on local system.'
            elif not local_data and remote_data:
                response = self.probe_partnership(local_data, remote_data)
                modify_remote = response[1]
                self.create_parameter_validation()
                self.create_partnership('local', self.remote_clusterip)
                msg += ' IP partnership created on local system.'
                if modify_remote:
                    self.update_partnership('remote', local_id, modify_remote)
                    msg += 'IP partnership updated on {0} system.'.format(['remote'])
                else:
                    msg += 'IP Partnership already exists on remote system.'
            elif not local_data and (not remote_data):
                self.create_parameter_validation()
                self.create_partnership('local', self.remote_clusterip)
                self.create_partnership('remote', local_ip)
                msg = 'IP partnership created on both local and remote system.'
        elif self.state == 'absent':
            self.delete_parameter_validation()
            if local_data and remote_data:
                self.remove_partnership('local', self.remote_cluster_id)
                self.remove_partnership('remote', local_id)
                msg += 'IP partnership deleted from both local and remote system.'
            elif local_data and (not remote_data):
                self.remove_partnership('local', self.remote_cluster_id)
                msg += 'IP partnership deleted from local system.'
                msg += ' IP partnership does not exists on remote system.'
            elif not local_data and remote_data:
                self.remove_partnership('remote', local_id)
                msg += 'IP partnership deleted from remote system.'
                msg += ' IP partnership does not exists on local system.'
            elif not local_data and (not remote_data):
                msg += 'IP partnership does not exists on both local and remote system. No modifications done.'
        if self.module.check_mode:
            msg = 'Skipping changes due to check mode.'
        self.module.exit_json(msg=msg, changed=self.changed)