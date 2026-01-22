from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
class IBMSVCUser(object):

    def __init__(self):
        argument_spec = svc_argument_spec()
        argument_spec.update(dict(name=dict(type='str', required=True), state=dict(type='str', required=True, choices=['present', 'absent']), auth_type=dict(type='str', required=False, choices=['usergrp']), user_password=dict(type='str', required=False, no_log=True), nopassword=dict(type='bool', required=False), keyfile=dict(type='str', required=False, no_log=True), nokey=dict(type='bool', required=False), forcepasswordchange=dict(type='bool', required=False), lock=dict(type='bool', required=False), unlock=dict(type='bool', required=False), usergroup=dict(type='str', required=False)))
        self.module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
        log_path = self.module.params['log_path']
        log = get_logger(self.__class__.__name__, log_path)
        self.log = log.info
        self.name = self.module.params['name']
        self.state = self.module.params['state']
        self.auth_type = self.module.params['auth_type']
        self.usergroup = self.module.params['usergroup']
        self.user_password = self.module.params.get('user_password', False)
        self.nopassword = self.module.params.get('nopassword', False)
        self.keyfile = self.module.params.get('keyfile', False)
        self.nokey = self.module.params.get('nokey', False)
        self.forcepasswordchange = self.module.params.get('forcepasswordchange', False)
        self.lock = self.module.params.get('lock', False)
        self.unlock = self.module.params.get('unlock', False)
        self.restapi = IBMSVCRestApi(module=self.module, clustername=self.module.params['clustername'], domain=self.module.params['domain'], username=self.module.params['username'], password=self.module.params['password'], validate_certs=self.module.params['validate_certs'], log_path=log_path, token=self.module.params['token'])

    def basic_checks(self):
        if not self.name:
            self.module.fail_json(msg='Missing mandatory parameter: name')
        if not self.state:
            self.module.fail_json(msg='Missing mandatory parameter: state')
        if self.user_password and self.nopassword:
            self.module.fail_json(msg='Mutually exclusive parameter: user_password, nopassword')
        if self.lock and self.unlock:
            self.module.fail_json(msg='Mutually exclusive parameter: lock, unlock')
        if self.keyfile and self.nokey:
            self.module.fail_json(msg='Mutually exclusive parameter: keyfile, nokey')
        if self.auth_type == 'usergrp' and (not self.usergroup):
            self.module.fail_json(msg='Parameter [usergroup] is required when auth_type is usergrp')

    def get_existing_user(self):
        merged_result = {}
        data = self.restapi.svc_obj_info(cmd='lsuser', cmdopts=None, cmdargs=[self.name])
        self.log('GET: user data: %s', data)
        if isinstance(data, list):
            for d in data:
                merged_result.update(d)
        else:
            merged_result = data
        return merged_result

    def create_user(self):
        if self.nokey or self.nopassword or self.lock or self.unlock or self.forcepasswordchange:
            self.module.fail_json(msg='Parameters [nokey, nopassword, lock, unlock, forcepasswordchange] not applicable while creating a user')
        if not self.auth_type:
            self.module.fail_json(msg='Missing required parameter: auth_type')
        if self.auth_type == 'usergrp' and (not self.usergroup):
            self.module.fail_json(msg='Missing required parameter: usergroup')
        if self.module.check_mode:
            self.changed = True
            return
        command = 'mkuser'
        command_options = {'name': self.name}
        if self.user_password:
            command_options['password'] = self.user_password
        if self.keyfile:
            command_options['keyfile'] = self.keyfile
        if self.usergroup:
            command_options['usergrp'] = self.usergroup
        if self.forcepasswordchange:
            command_options['forcepasswordchange'] = self.forcepasswordchange
        result = self.restapi.svc_run_command(command, command_options, cmdargs=None)
        self.log('create user result %s', result)
        if 'message' in result:
            self.changed = True
            self.log('create user result message %s', result['message'])
        else:
            self.module.fail_json(msg='Failed to create user [%s]' % self.name)

    def probe_user(self, data):
        properties = {}
        if self.usergroup:
            if self.usergroup != data['usergrp_name']:
                properties['usergrp'] = self.usergroup
        if self.user_password:
            properties['password'] = self.user_password
        if self.nopassword:
            if data['password'] == 'yes':
                properties['nopassword'] = True
        if self.keyfile:
            properties['keyfile'] = self.keyfile
        if self.nokey:
            if data['ssh_key'] == 'yes':
                properties['nokey'] = True
        if self.lock:
            properties['lock'] = True
        if self.unlock:
            properties['unlock'] = True
        if self.forcepasswordchange:
            properties['forcepasswordchange'] = True
        return properties

    def update_user(self, data):
        if self.module.check_mode:
            self.changed = True
            return
        self.log("updating user '%s'", self.name)
        command = 'chuser'
        for parameter in data:
            command_options = {parameter: data[parameter]}
            self.restapi.svc_run_command(command, command_options, [self.name])
        self.changed = True

    def remove_user(self):
        if self.nokey or self.nopassword or self.lock or self.unlock or self.forcepasswordchange:
            self.module.fail_json(msg='Parameters [nokey, nopassword, lock, unlock, forcepasswordchange] not applicable while removing a user')
        if self.module.check_mode:
            self.changed = True
            return
        self.log("deleting user '%s'", self.name)
        command = 'rmuser'
        command_options = None
        cmdargs = [self.name]
        self.restapi.svc_run_command(command, command_options, cmdargs)
        self.changed = True

    def apply(self):
        changed = False
        msg = None
        modify = {}
        self.basic_checks()
        user_data = self.get_existing_user()
        if user_data:
            if self.state == 'absent':
                self.log("CHANGED: user exists, but requested state is 'absent'")
                changed = True
            elif self.state == 'present':
                modify = self.probe_user(user_data)
                if modify:
                    self.log('CHANGED: user exists, but probe detected changes')
                    changed = True
        elif self.state == 'present':
            self.log("CHANGED: user does not exist, but requested state is 'present'")
            changed = True
        if changed:
            if self.state == 'present':
                if not user_data:
                    self.create_user()
                    msg = 'User [%s] has been created.' % self.name
                else:
                    self.update_user(modify)
                    msg = 'User [%s] has been modified.' % self.name
            elif self.state == 'absent':
                self.remove_user()
                msg = 'User [%s] has been removed.' % self.name
            if self.module.check_mode:
                msg = 'Skipping changes due to check mode.'
        elif self.state == 'absent':
            msg = 'User [%s] does not exist.' % self.name
        elif self.state == 'present':
            msg = 'User [%s] already exist (no modificationes detected).' % self.name
        self.module.exit_json(msg=msg, changed=changed)