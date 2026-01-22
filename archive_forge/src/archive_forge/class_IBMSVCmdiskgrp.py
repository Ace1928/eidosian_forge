from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
class IBMSVCmdiskgrp(object):

    def __init__(self):
        argument_spec = svc_argument_spec()
        argument_spec.update(dict(name=dict(type='str', required=True), state=dict(type='str', required=True, choices=['absent', 'present']), datareduction=dict(type='str', default='no', choices=['yes', 'no']), easytier=dict(type='str', default='off', choices=['on', 'off', 'auto']), encrypt=dict(type='str', default='no', choices=['yes', 'no']), ext=dict(type='int'), parentmdiskgrp=dict(type='str'), safeguarded=dict(type='bool'), noquota=dict(type='bool'), size=dict(type='int'), unit=dict(type='str'), provisioningpolicy=dict(type='str'), noprovisioningpolicy=dict(type='bool'), replicationpoollinkuid=dict(type='str'), resetreplicationpoollinkuid=dict(type='bool'), replication_partner_clusterid=dict(type='str'), warning=dict(type='int'), vdiskprotectionenabled=dict(type='str', choices=['yes', 'no']), ownershipgroup=dict(type='str'), noownershipgroup=dict(type='bool'), etfcmoverallocationmax=dict(type='str'), old_name=dict(type='str')))
        mutually_exclusive = []
        self.module = AnsibleModule(argument_spec=argument_spec, mutually_exclusive=mutually_exclusive, supports_check_mode=True)
        log_path = self.module.params['log_path']
        log = get_logger(self.__class__.__name__, log_path)
        self.log = log.info
        self.name = self.module.params['name']
        self.state = self.module.params['state']
        self.datareduction = self.module.params.get('datareduction', None)
        self.easytier = self.module.params.get('easytier', None)
        self.encrypt = self.module.params.get('encrypt', None)
        self.ext = self.module.params.get('ext', None)
        self.safeguarded = self.module.params.get('safeguarded', False)
        self.noquota = self.module.params.get('noquota', False)
        self.provisioningpolicy = self.module.params.get('provisioningpolicy', '')
        self.noprovisioningpolicy = self.module.params.get('noprovisioningpolicy', False)
        self.replicationpoollinkuid = self.module.params.get('replicationpoollinkuid', '')
        self.resetreplicationpoollinkuid = self.module.params.get('resetreplicationpoollinkuid', False)
        self.replication_partner_clusterid = self.module.params.get('replication_partner_clusterid', '')
        self.warning = self.module.params.get('warning', None)
        self.ownershipgroup = self.module.params.get('ownershipgroup', '')
        self.noownershipgroup = self.module.params.get('noownershipgroup', False)
        self.vdiskprotectionenabled = self.module.params.get('vdiskprotectionenabled', None)
        self.etfcmoverallocationmax = self.module.params.get('etfcmoverallocationmax', '')
        self.old_name = self.module.params.get('old_name', '')
        self.parentmdiskgrp = self.module.params.get('parentmdiskgrp', None)
        self.size = self.module.params.get('size', None)
        self.unit = self.module.params.get('unit', None)
        self.changed = False
        self.partnership_index = None
        self.basic_checks()
        self.restapi = IBMSVCRestApi(module=self.module, clustername=self.module.params['clustername'], domain=self.module.params['domain'], username=self.module.params['username'], password=self.module.params['password'], validate_certs=self.module.params['validate_certs'], log_path=log_path, token=self.module.params['token'])

    def basic_checks(self):
        if not self.name:
            self.module.fail_json(msg='Missing mandatory parameter: name')
        if self.state == 'present':
            message = 'Following parameters are required together: replicationpoollinkuid, replication_partner_clusterid'
            if self.replication_partner_clusterid:
                if not self.replicationpoollinkuid:
                    self.module.fail_json(msg=message)
            elif self.replicationpoollinkuid:
                self.module.fail_json(msg=message)
            if self.replicationpoollinkuid and self.resetreplicationpoollinkuid:
                self.module.fail_json(msg='Mutually exclusive parameters: replicationpoollinkuid, resetreplicationpoollinkuid')
        elif self.state == 'absent':
            invalids = ('warning', 'ownershipgroup', 'noownershipgroup', 'vdiskprotectionenabled', 'etfcmoverallocationmax', 'old_name')
            invalid_exists = ', '.join((var for var in invalids if getattr(self, var) not in {'', None}))
            if invalid_exists:
                self.module.fail_json(msg='state=absent but following parameters have been passed: {0}'.format(invalid_exists))

    def create_validation(self):
        invalids = ('noownershipgroup', 'old_name')
        invalid_exists = ', '.join((var for var in invalids if getattr(self, var) not in {'', None}))
        if invalid_exists:
            self.module.fail_json(msg='Following parameters not supported during creation: {0}'.format(invalid_exists))

    def mdiskgrp_rename(self, mdiskgrp_data):
        msg = None
        old_mdiskgrp_data = self.mdiskgrp_exists(self.old_name)
        if not old_mdiskgrp_data and (not mdiskgrp_data):
            self.module.fail_json(msg='mdiskgrp [{0}] does not exists.'.format(self.old_name))
        elif old_mdiskgrp_data and mdiskgrp_data:
            self.module.fail_json(msg='mdiskgrp with name [{0}] already exists.'.format(self.name))
        elif not old_mdiskgrp_data and mdiskgrp_data:
            msg = 'mdiskgrp [{0}] already renamed.'.format(self.name)
        elif old_mdiskgrp_data and (not mdiskgrp_data):
            if self.old_name == self.parentmdiskgrp:
                self.module.fail_json("Old name shouldn't be same as parentmdiskgrp while renaming childmdiskgrp")
            if self.module.check_mode:
                self.changed = True
                return
            self.restapi.svc_run_command('chmdiskgrp', {'name': self.name}, [self.old_name])
            self.changed = True
            msg = 'mdiskgrp [{0}] has been successfully rename to [{1}].'.format(self.old_name, self.name)
        return msg

    def mdiskgrp_exists(self, name):
        merged_result = {}
        data = self.restapi.svc_obj_info(cmd='lsmdiskgrp', cmdopts=None, cmdargs=['-gui', name])
        if isinstance(data, list):
            for d in data:
                merged_result.update(d)
        else:
            merged_result = data
        return merged_result

    def mdiskgrp_create(self):
        self.create_validation()
        self.log("creating mdisk group '%s'", self.name)
        cmd = 'mkmdiskgrp'
        cmdopts = {}
        if not self.ext:
            self.module.fail_json(msg='You must pass the ext to the module.')
        if self.noquota or self.safeguarded:
            if not self.parentmdiskgrp:
                self.module.fail_json(msg='Required parameter missing: parentmdiskgrp')
        self.check_partnership()
        if self.module.check_mode:
            self.changed = True
            return
        if self.parentmdiskgrp:
            cmdopts['parentmdiskgrp'] = self.parentmdiskgrp
            if self.size:
                cmdopts['size'] = self.size
            if self.unit:
                cmdopts['unit'] = self.unit
            if self.safeguarded:
                cmdopts['safeguarded'] = self.safeguarded
            if self.noquota:
                cmdopts['noquota'] = self.noquota
        else:
            if self.easytier:
                cmdopts['easytier'] = self.easytier
            if self.encrypt:
                cmdopts['encrypt'] = self.encrypt
            if self.ext:
                cmdopts['ext'] = str(self.ext)
        if self.provisioningpolicy:
            cmdopts['provisioningpolicy'] = self.provisioningpolicy
        if self.datareduction:
            cmdopts['datareduction'] = self.datareduction
        if self.replicationpoollinkuid:
            cmdopts['replicationpoollinkuid'] = self.replicationpoollinkuid
        if self.ownershipgroup:
            cmdopts['ownershipgroup'] = self.ownershipgroup
        if self.vdiskprotectionenabled:
            cmdopts['vdiskprotectionenabled'] = self.vdiskprotectionenabled
        if self.etfcmoverallocationmax:
            if '%' not in self.etfcmoverallocationmax and self.etfcmoverallocationmax != 'off':
                cmdopts['etfcmoverallocationmax'] = self.etfcmoverallocationmax + '%'
            else:
                cmdopts['etfcmoverallocationmax'] = self.etfcmoverallocationmax
        if self.warning:
            cmdopts['warning'] = str(self.warning) + '%'
        cmdopts['name'] = self.name
        self.log('creating mdisk group command %s opts %s', cmd, cmdopts)
        result = self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
        self.log('creating mdisk group result %s', result)
        if self.replication_partner_clusterid:
            self.set_bit_mask()
        if 'message' in result:
            self.log('creating mdisk group command result message %s', result['message'])
        else:
            self.module.fail_json(msg='Failed to create mdisk group [%s]' % self.name)

    def check_partnership(self):
        if self.replication_partner_clusterid:
            merged_result = {}
            result = self.restapi.svc_obj_info(cmd='lspartnership', cmdopts=None, cmdargs=['-gui', self.replication_partner_clusterid])
            if isinstance(result, list):
                for res in result:
                    merged_result = res
            else:
                merged_result = result
            if merged_result:
                self.partnership_index = merged_result.get('partnership_index')
            else:
                self.module.fail_json(msg='Partnership does not exist for the given cluster ({0}).'.format(self.replication_partner_clusterid))

    def set_bit_mask(self, systemmask=None):
        cmd = 'chmdiskgrp'
        bit_mask = '1'.ljust(int(self.partnership_index) + 1, '0') if not systemmask else systemmask
        cmdopts = {'replicationpoollinkedsystemsmask': bit_mask}
        self.restapi.svc_run_command(cmd, cmdopts, cmdargs=[self.name])

    def mdiskgrp_delete(self):
        if self.module.check_mode:
            self.changed = True
            return
        self.log("deleting mdiskgrp '%s'", self.name)
        cmd = 'rmmdiskgrp'
        cmdopts = None
        cmdargs = [self.name]
        self.restapi.svc_run_command(cmd, cmdopts, cmdargs)

    def mdiskgrp_update(self, modify):
        self.log("updating mdiskgrp '%s'", self.name)
        systemmask = None
        cmd = 'chmdiskgrp'
        if 'replicationpoollinkedsystemsmask' in modify:
            systemmask = modify.pop('replicationpoollinkedsystemsmask')
        if modify:
            cmdopts = modify
            self.restapi.svc_run_command(cmd, cmdopts, cmdargs=[self.name])
        if systemmask or 'replicationpoollinkuid' in modify:
            self.set_bit_mask(systemmask)
        self.changed = True

    def mdiskgrp_probe(self, data):
        props = {}
        if self.noprovisioningpolicy and data.get('provisioning_policy_name', ''):
            props['noprovisioningpolicy'] = self.noprovisioningpolicy
        if self.provisioningpolicy and self.provisioningpolicy != data.get('provisioning_policy_name', ''):
            props['provisioningpolicy'] = self.provisioningpolicy
        if self.noownershipgroup and data.get('owner_name', ''):
            props['noownershipgroup'] = self.noownershipgroup
        if self.ownershipgroup and self.ownershipgroup != data.get('owner_name', ''):
            props['ownershipgroup'] = self.ownershipgroup
        if self.vdiskprotectionenabled and self.vdiskprotectionenabled != data.get('vdisk_protectionenabled', ''):
            props['vdiskprotectionenabled'] = self.vdiskprotectionenabled
        if self.warning and self.warning != data.get('warning', ''):
            props['warning'] = str(self.warning) + '%'
        if self.replicationpoollinkuid and self.replicationpoollinkuid != data.get('replication_pool_link_uid', ''):
            props['replicationpoollinkuid'] = self.replicationpoollinkuid
        if self.resetreplicationpoollinkuid:
            props['resetreplicationpoollinkuid'] = self.resetreplicationpoollinkuid
        if self.etfcmoverallocationmax:
            if '%' not in self.etfcmoverallocationmax and self.etfcmoverallocationmax != 'off':
                self.etfcmoverallocationmax += '%'
            if self.etfcmoverallocationmax != data.get('easy_tier_fcm_over_allocation_max', ''):
                props['etfcmoverallocationmax'] = self.etfcmoverallocationmax
        if self.replication_partner_clusterid:
            self.check_partnership()
            bit_mask = '1'.ljust(int(self.partnership_index) + 1, '0')
            if bit_mask.zfill(64) != data.get('replication_pool_linked_systems_mask', ''):
                props['replicationpoollinkedsystemsmask'] = bit_mask
        self.log("mdiskgrp_probe props='%s'", props)
        return props

    def apply(self):
        changed = False
        msg = None
        modify = []
        mdiskgrp_data = self.mdiskgrp_exists(self.name)
        if self.state == 'present' and self.old_name:
            msg = self.mdiskgrp_rename(mdiskgrp_data)
        elif self.state == 'absent' and self.old_name:
            self.module.fail_json(msg="Rename functionality is not supported when 'state' is absent.")
        else:
            if mdiskgrp_data:
                if self.state == 'absent':
                    self.log("CHANGED: mdisk group exists, but requested state is 'absent'")
                    changed = True
                elif self.state == 'present':
                    modify = self.mdiskgrp_probe(mdiskgrp_data)
                    if modify:
                        changed = True
            elif self.state == 'present':
                self.log("CHANGED: mdisk group does not exist, but requested state is 'present'")
                changed = True
            if changed:
                if self.state == 'present':
                    if not mdiskgrp_data:
                        self.mdiskgrp_create()
                        self.changed = True
                        msg = 'Mdisk group [%s] has been created.' % self.name
                    else:
                        self.mdiskgrp_update(modify)
                        msg = 'Mdisk group [%s] has been modified.' % self.name
                elif self.state == 'absent':
                    self.mdiskgrp_delete()
                    self.changed = True
                    msg = 'mdiskgrp [%s] has been deleted.' % self.name
            else:
                self.log('exiting with no changes')
                if self.state == 'absent':
                    msg = 'Mdisk group [%s] did not exist.' % self.name
                else:
                    msg = 'Mdisk group [%s] already exists. No modifications done' % self.name
        if self.module.check_mode:
            msg = 'skipping changes due to check mode'
        self.module.exit_json(msg=msg, changed=self.changed)