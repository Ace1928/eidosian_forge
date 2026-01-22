from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
class IBMSVCFlashcopyConsistgrp(object):

    def __init__(self):
        argument_spec = svc_argument_spec()
        argument_spec.update(dict(name=dict(type='str', required=True), state=dict(type='str', required=True, choices=['present', 'absent']), ownershipgroup=dict(type='str', required=False), noownershipgroup=dict(type='bool', required=False), force=dict(type='bool', required=False)))
        self.module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
        log_path = self.module.params['log_path']
        log = get_logger(self.__class__.__name__, log_path)
        self.log = log.info
        self.name = self.module.params['name']
        self.state = self.module.params['state']
        self.ownershipgroup = self.module.params.get('ownershipgroup', False)
        self.noownershipgroup = self.module.params.get('noownershipgroup', False)
        self.force = self.module.params.get('force', False)
        if not self.name:
            self.module.fail_json(msg='Missing mandatory parameter: name')
        self.restapi = IBMSVCRestApi(module=self.module, clustername=self.module.params['clustername'], domain=self.module.params['domain'], username=self.module.params['username'], password=self.module.params['password'], validate_certs=self.module.params['validate_certs'], log_path=log_path, token=self.module.params['token'])

    def get_existing_fcconsistgrp(self):
        data = {}
        data = self.restapi.svc_obj_info(cmd='lsfcconsistgrp', cmdopts=None, cmdargs=[self.name])
        return data

    def fcconsistgrp_create(self):
        if self.module.check_mode:
            self.changed = True
            return
        cmd = 'mkfcconsistgrp'
        cmdopts = {}
        cmdopts['name'] = self.name
        if self.ownershipgroup:
            cmdopts['ownershipgroup'] = self.ownershipgroup
        self.log('Creating fc consistgrp.. Command: %s opts %s', cmd, cmdopts)
        result = self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
        if 'message' in result:
            self.changed = True
            self.log('Create fc consistgrp message %s', result['message'])
        else:
            self.module.fail_json(msg='Failed to create fc consistgrp [%s]' % self.name)

    def fcconsistgrp_delete(self):
        if self.module.check_mode:
            self.changed = True
            return
        cmd = 'rmfcconsistgrp'
        cmdopts = {}
        if self.force:
            cmdopts['force'] = self.force
        self.log('Deleting fc consistgrp.. Command %s opts %s', cmd, cmdopts)
        self.restapi.svc_run_command(cmd, cmdopts, cmdargs=[self.name])

    def fcconsistgrp_probe(self, data):
        props = {}
        self.log('Probe which properties need to be updated...')
        if not self.noownershipgroup:
            if self.ownershipgroup and self.ownershipgroup != data['owner_name']:
                props['ownershipgroup'] = self.ownershipgroup
        if self.noownershipgroup and data['owner_name']:
            props['noownershipgroup'] = self.noownershipgroup
        return props

    def fcconsistgrp_update(self, modify):
        if self.module.check_mode:
            self.changed = True
            return
        if modify:
            self.log('updating fcmap with properties %s', modify)
            cmd = 'chfcconsistgrp'
            cmdopts = {}
            for prop in modify:
                cmdopts[prop] = modify[prop]
            cmdargs = [self.name]
            self.restapi.svc_run_command(cmd, cmdopts, cmdargs)

    def apply(self):
        changed = False
        msg = None
        modify = []
        gdata = self.get_existing_fcconsistgrp()
        if gdata:
            if self.state == 'absent':
                self.log("fc consistgrp [%s] exist, but requested state is 'absent'", self.name)
                changed = True
            elif self.state == 'present':
                modify = self.fcconsistgrp_probe(gdata)
                if modify:
                    changed = True
        elif self.state == 'present':
            self.log("fc consistgrp [%s] doesn't exist, but requested state is 'present'", self.name)
            changed = True
        if changed:
            if self.state == 'absent':
                self.fcconsistgrp_delete()
                msg = 'fc consistgrp [%s] has been deleted' % self.name
            elif self.state == 'present' and modify:
                self.fcconsistgrp_update(modify)
                msg = 'fc consistgrp [%s] has been modified' % self.name
            elif self.state == 'present' and (not modify):
                self.fcconsistgrp_create()
                msg = 'fc consistgrp [%s] has been created' % self.name
            if self.module.check_mode:
                msg = 'skipping changes due to check mode.'
        elif self.state == 'absent':
            msg = 'fc consistgrp [%s] does not exist' % self.name
        elif self.state == 'present':
            msg = 'fc consistgrp [%s] already exists' % self.name
        self.module.exit_json(msg=msg, changed=changed)