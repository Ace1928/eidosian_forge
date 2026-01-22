from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import \
from ansible.module_utils._text import to_native
class IBMSVCRCCG(object):

    def __init__(self):
        argument_spec = svc_argument_spec()
        argument_spec.update(dict(name=dict(type='str', required=True), state=dict(type='str', required=True, choices=['absent', 'present']), remotecluster=dict(type='str', required=False), force=dict(type='bool', required=False), copytype=dict(type='str', choices=['metro', 'global']), cyclingmode=dict(type='str', required=False, choices=['multi', 'none']), cyclingperiod=dict(type='int', required=False)))
        self.module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
        log_path = self.module.params['log_path']
        log = get_logger(self.__class__.__name__, log_path)
        self.log = log.info
        self.name = self.module.params['name']
        self.state = self.module.params['state']
        self.cluster = self.module.params.get('remotecluster', None)
        self.force = self.module.params.get('force', False)
        self.copytype = self.module.params.get('copytype', None)
        self.cyclingmode = self.module.params.get('cyclingmode', None)
        self.cyclingperiod = self.module.params.get('cyclingperiod', None)
        if not self.name:
            self.module.fail_json(msg='Missing mandatory parameter: name')
        self.restapi = IBMSVCRestApi(module=self.module, clustername=self.module.params['clustername'], domain=self.module.params['domain'], username=self.module.params['username'], password=self.module.params['password'], validate_certs=self.module.params['validate_certs'], log_path=log_path, token=self.module.params['token'])

    def get_existing_rccg(self):
        merged_result = {}
        data = self.restapi.svc_obj_info(cmd='lsrcconsistgrp', cmdopts=None, cmdargs=[self.name])
        if isinstance(data, list):
            for d in data:
                merged_result.update(d)
        else:
            merged_result = data
        return merged_result

    def rccg_probe(self, data):
        props = {}
        propscv = {}
        if self.copytype and self.copytype != data['copy_type']:
            if self.copytype == 'global':
                props['global'] = True
            elif self.copytype == 'metro':
                props['metro'] = True
            else:
                self.module.fail_json(msg="Unsupported mirror type: %s. Only 'global' and 'metro' are supported when modifying" % self.copytype)
        if self.copytype == 'global' and self.cyclingperiod and (self.cyclingperiod != int(data['cycle_period_seconds'])):
            propscv['cycleperiodseconds'] = self.cyclingperiod
        if self.copytype == 'global' and self.cyclingmode and (self.cyclingmode != data['cycling_mode']):
            propscv['cyclingmode'] = self.cyclingmode
        return (props, propscv)

    def rccg_create(self):
        if self.module.check_mode:
            self.changed = True
            return
        rccg_data = self.get_existing_rccg()
        if rccg_data:
            self.rccg_update(rccg_data)
        self.log("creating rc consistgrp '%s'", self.name)
        cmd = 'mkrcconsistgrp'
        cmdopts = {'name': self.name}
        if self.cluster:
            cmdopts['cluster'] = self.cluster
        self.log("creating rc consistgrp command '%s' opts", self.cluster)
        result = self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
        self.log("create rc consistgrp result '%s'", result)
        msg = "succeeded to create rc consistgrp '%s'" % self.name
        self.log(msg)
        if 'message' in result:
            self.log("create rc consistgrp result message '%s'", result['message'])
            self.module.exit_json(msg="rc consistgrp '%s' is created" % self.name, changed=True)
        else:
            self.module.fail_json(msg=result)

    def rccg_update(self, modify, modifycv):
        if modify:
            self.log('updating chrcconsistgrp with properties %s', modify)
            cmd = 'chrcconsistgrp'
            cmdopts = {}
            for prop in modify:
                cmdopts[prop] = modify[prop]
            cmdargs = [self.name]
            self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
            self.changed = True
        if modifycv:
            self.log('updating chrcconsistgrp with properties %s', modifycv)
            cmd = 'chrcconsistgrp'
            cmdargs = [self.name]
            for prop in modifycv:
                cmdopts = {}
                cmdopts[prop] = modifycv[prop]
                self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
            self.changed = True
        if not modify and (not modifycv):
            self.log('There is no property to be updated')
            self.changed = False

    def rccg_delete(self):
        rccg_data = self.get_existing_rccg()
        if not rccg_data:
            self.module.exit_json(msg="rc consistgrp '%s' did not exist" % self.name, changed=False)
        if self.module.check_mode:
            self.changed = True
            return
        self.log("deleting rc consistgrp '%s'", self.name)
        cmd = 'rmrcconsistgrp'
        cmdopts = {'force': True} if self.force else None
        cmdargs = [self.name]
        self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
        msg = "rc consistgrp '%s' is deleted" % self.name
        self.log(msg)
        self.module.exit_json(msg=msg, changed=True)

    def apply(self):
        changed = False
        msg = None
        modify = {}
        rccg_data = self.get_existing_rccg()
        if rccg_data:
            if self.state == 'absent':
                self.log("CHANGED: RemoteCopy group exists, requested state is 'absent'")
                changed = True
            elif self.state == 'present':
                modify, modifycv = self.rccg_probe(rccg_data)
                if modify or modifycv:
                    changed = True
        elif self.state == 'present':
            if self.copytype:
                self.module.fail_json(msg='copytype cannot be passed while creating a consistency group')
            changed = True
            self.log("CHANGED: Remotecopy group does not exist, but requested state is '%s'", self.state)
        if changed:
            if self.state == 'present':
                if not rccg_data:
                    self.rccg_create()
                    msg = 'remote copy group %s has been created.' % self.name
                else:
                    self.rccg_update(modify, modifycv)
                    msg = 'remote copy group [%s] has been modified.' % self.name
            elif self.state == 'absent':
                self.rccg_delete()
                msg = 'remote copy group [%s] has been deleted.' % self.name
            if self.module.check_mode:
                msg = 'skipping changes due to check mode.'
        else:
            self.log('exiting with no changes')
            if self.state in ['absent']:
                msg = 'Remotecopy group [%s] does not exist.' % self.name
            else:
                msg = 'No Modifications detected, Remotecopy group [%s] already exists.' % self.name
        self.module.exit_json(msg=msg, changed=changed)