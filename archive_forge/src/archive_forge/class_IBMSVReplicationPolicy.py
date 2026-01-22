from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
class IBMSVReplicationPolicy:

    def __init__(self):
        argument_spec = svc_argument_spec()
        argument_spec.update(dict(name=dict(type='str', required=True), state=dict(type='str', choices=['present', 'absent'], required=True), topology=dict(type='str', choices=['2-site-async-dr']), location1system=dict(type='str'), location1iogrp=dict(type='int'), location2system=dict(type='str'), location2iogrp=dict(type='int'), rpoalert=dict(type='int')))
        self.module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
        self.name = self.module.params['name']
        self.state = self.module.params['state']
        self.topology = self.module.params.get('topology', '')
        self.location1system = self.module.params.get('location1system', '')
        self.location1iogrp = self.module.params.get('location1iogrp', '')
        self.location2system = self.module.params.get('location2system', '')
        self.location2iogrp = self.module.params.get('location2iogrp', '')
        self.rpoalert = self.module.params.get('rpoalert', '')
        self.basic_checks()
        self.log_path = self.module.params['log_path']
        log = get_logger(self.__class__.__name__, self.log_path)
        self.log = log.info
        self.changed = False
        self.msg = ''
        self.rp_data = {}
        self.restapi = IBMSVCRestApi(module=self.module, clustername=self.module.params['clustername'], domain=self.module.params['domain'], username=self.module.params['username'], password=self.module.params['password'], validate_certs=self.module.params['validate_certs'], log_path=self.log_path, token=self.module.params['token'])

    def basic_checks(self):
        if not self.name:
            self.module.fail_json(msg='Missing mandatory parameter: name')
        if self.state == 'absent':
            invalids = ('topology', 'location1system', 'location1iogrp', 'location2system', 'location2iogrp', 'rpoalert')
            invalid_exists = ', '.join((var for var in invalids if not getattr(self, var) in {'', None}))
            if invalid_exists:
                self.module.fail_json(msg='state=absent but following paramters have been passed: {0}'.format(invalid_exists))

    def is_rp_exists(self):
        result = {}
        cmd = 'lsreplicationpolicy'
        data = self.restapi.svc_obj_info(cmd=cmd, cmdopts=None, cmdargs=[self.name])
        if isinstance(data, list):
            for d in data:
                result.update(d)
        else:
            result = data
        self.rp_data = result
        return result

    def create_replication_policy(self):
        if self.module.check_mode:
            self.changed = True
            return
        cmd = 'mkreplicationpolicy'
        cmdopts = {'name': self.name, 'topology': self.topology, 'location1system': self.location1system, 'location1iogrp': self.location1iogrp, 'location2system': self.location2system, 'location2iogrp': self.location2iogrp, 'rpoalert': self.rpoalert}
        self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
        self.log('Replication policy (%s) created', self.name)
        self.changed = True

    def replication_policy_probe(self):
        field_mappings = (('topology', self.rp_data.get('topology', '')), ('location1system', (('location1_system_name', self.rp_data.get('location1_system_name', '')), ('location1_system_id', self.rp_data.get('location1_system_id', '')))), ('location1iogrp', self.rp_data.get('location1_iogrp_id', '')), ('location2system', (('location2_system_name', self.rp_data.get('location2_system_name', '')), ('location2_system_id', self.rp_data.get('location2_system_id', '')))), ('location2iogrp', self.rp_data.get('location2_iogrp_id', '')), ('rpoalert', self.rp_data.get('rpo_alert', '')))
        self.log('replication policy probe data: %s', field_mappings)
        for f, v in field_mappings:
            current_value = str(getattr(self, f))
            if current_value and f in {'location1system', 'location2system'}:
                try:
                    next(iter(filter(lambda val: val[1] == current_value, v)))
                except StopIteration:
                    self.module.fail_json(msg='Policy modification is not supported. Please delete and recreate new policy.')
            elif current_value and current_value != v:
                self.module.fail_json(msg='Policy modification is not supported. Please delete and recreate new policy.')

    def delete_replication_policy(self):
        if self.module.check_mode:
            self.changed = True
            return
        cmd = 'rmreplicationpolicy'
        self.restapi.svc_run_command(cmd, cmdopts=None, cmdargs=[self.name])
        self.log('Replication policy (%s) deleted', self.name)
        self.changed = True

    def apply(self):
        if self.is_rp_exists():
            if self.state == 'present':
                self.replication_policy_probe()
                self.msg = 'Replication policy ({0}) already exists. No modifications done.'.format(self.name)
            else:
                self.delete_replication_policy()
                self.msg = 'Replication policy ({0}) deleted'.format(self.name)
        elif self.state == 'absent':
            self.msg = 'Replication policy ({0}) does not exists.'.format(self.name)
        else:
            self.create_replication_policy()
            self.msg = 'Replication policy ({0}) created.'.format(self.name)
        if self.module.check_mode:
            self.msg = 'skipping changes due to check mode.'
        self.module.exit_json(changed=self.changed, msg=self.msg)