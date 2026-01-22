from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
class IBMSVCUsergroup(object):

    def __init__(self):
        argument_spec = svc_argument_spec()
        argument_spec.update(dict(name=dict(type='str', required=True), role=dict(type='str', required=False, choices=['Monitor', 'CopyOperator', 'Service', 'FlashCopyAdmin', 'Administrator', 'SecurityAdmin', 'VasaProvider', 'RestrictedAdmin', '3SiteAdmin']), ownershipgroup=dict(type='str', required=False), noownershipgroup=dict(type='bool', required=False), state=dict(type='str', required=True, choices=['present', 'absent'])))
        self.module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
        log_path = self.module.params['log_path']
        log = get_logger(self.__class__.__name__, log_path)
        self.log = log.info
        self.name = self.module.params['name']
        self.state = self.module.params['state']
        self.role = self.module.params['role']
        self.ownershipgroup = self.module.params.get('ownershipgroup', False)
        self.noownershipgroup = self.module.params.get('noownershipgroup', False)
        self.restapi = IBMSVCRestApi(module=self.module, clustername=self.module.params['clustername'], domain=self.module.params['domain'], username=self.module.params['username'], password=self.module.params['password'], validate_certs=self.module.params['validate_certs'], log_path=log_path, token=self.module.params['token'])

    def basic_checks(self):
        if not self.name:
            self.module.fail_json(msg='Missing mandatory parameter: name')
        if not self.state:
            self.module.fail_json(msg='Missing mandatory parameter: state')
        if self.ownershipgroup and self.noownershipgroup:
            self.module.fail_json(msg='Mutually exclusive parameter: ownershipgroup, noownershipgroup')
        if self.state == 'absent' and (self.role or self.ownershipgroup or self.noownershipgroup):
            self.module.fail_json(msg='Parameters [role, ownershipgroup, noownershipgroup] are not applicable while removing a usergroup')

    def get_existing_usergroup(self):
        merged_result = {}
        data = self.restapi.svc_obj_info(cmd='lsusergrp', cmdopts=None, cmdargs=[self.name])
        self.log('GET: user group data: %s', data)
        if isinstance(data, list):
            for d in data:
                merged_result.update(d)
        else:
            merged_result = data
        return merged_result

    def create_user_group(self):
        if self.noownershipgroup:
            self.module.fail_json(msg='Parameter [noownershipgroup] is not applicable while creating a usergroup')
        if not self.role:
            self.module.fail_json(msg='Missing mandatory parameter: role')
        if self.module.check_mode:
            self.changed = True
            return
        command = 'mkusergrp'
        command_options = {'name': self.name}
        if self.role:
            command_options['role'] = self.role
        if self.ownershipgroup:
            command_options['ownershipgroup'] = self.ownershipgroup
        result = self.restapi.svc_run_command(command, command_options, cmdargs=None)
        self.log('create user group result %s', result)
        if 'message' in result:
            self.changed = True
            self.log('create user group result message %s', result['message'])
        else:
            self.module.fail_json(msg='Failed to user volume group [%s]' % self.name)

    def probe_user_group(self, data):
        properties = {}
        if self.role:
            if self.role != data['role']:
                properties['role'] = self.role
        if self.ownershipgroup:
            if self.ownershipgroup != data['owner_name']:
                properties['ownershipgroup'] = self.ownershipgroup
        if self.noownershipgroup:
            if data['owner_name']:
                properties['noownershipgroup'] = True
        return properties

    def update_user_group(self, data):
        if self.module.check_mode:
            self.changed = True
            return
        self.log("updating user group '%s'", self.name)
        command = 'chusergrp'
        command_options = {}
        if 'role' in data:
            command_options['role'] = data['role']
        if 'ownershipgroup' in data:
            command_options['ownershipgroup'] = data['ownershipgroup']
        if 'noownershipgroup' in data:
            command_options['noownershipgroup'] = True
        cmdargs = [self.name]
        self.restapi.svc_run_command(command, command_options, cmdargs)
        self.changed = True

    def remove_user_group(self):
        if self.module.check_mode:
            self.changed = True
            return
        self.log("deleting user group '%s'", self.name)
        command = 'rmusergrp'
        command_options = None
        cmdargs = [self.name]
        self.restapi.svc_run_command(command, command_options, cmdargs)
        self.changed = True

    def apply(self):
        changed = False
        msg = None
        modify = {}
        self.basic_checks()
        user_group_data = self.get_existing_usergroup()
        if user_group_data:
            if self.state == 'absent':
                self.log("CHANGED: user group exists, but requested state is 'absent'")
                changed = True
            elif self.state == 'present':
                modify = self.probe_user_group(user_group_data)
                if modify:
                    self.log('CHANGED: user group exists, but probe detected changes')
                    changed = True
        elif self.state == 'present':
            self.log("CHANGED: user group does not exist, but requested state is 'present'")
            changed = True
        if changed:
            if self.state == 'present':
                if not user_group_data:
                    self.create_user_group()
                    msg = 'User group [%s] has been created.' % self.name
                else:
                    self.update_user_group(modify)
                    msg = 'User group [%s] has been modified.' % self.name
            elif self.state == 'absent':
                self.remove_user_group()
                msg = 'User group [%s] has been removed.' % self.name
            if self.module.check_mode:
                msg = 'Skipping changes due to check mode.'
        elif self.state == 'absent':
            msg = 'User group [%s] does not exist.' % self.name
        elif self.state == 'present':
            msg = 'User group [%s] already exist (no modificationes detected).' % self.name
        self.module.exit_json(msg=msg, changed=changed)