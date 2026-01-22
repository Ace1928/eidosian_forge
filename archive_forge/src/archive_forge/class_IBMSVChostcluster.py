from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
class IBMSVChostcluster(object):

    def __init__(self):
        argument_spec = svc_argument_spec()
        argument_spec.update(dict(name=dict(type='str', required=True), state=dict(type='str', required=True, choices=['absent', 'present']), ownershipgroup=dict(type='str'), noownershipgroup=dict(type='bool'), removeallhosts=dict(type='bool')))
        self.changed = ''
        self.module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
        log_path = self.module.params['log_path']
        log = get_logger(self.__class__.__name__, log_path)
        self.log = log.info
        self.name = self.module.params['name']
        self.state = self.module.params['state']
        self.ownershipgroup = self.module.params.get('ownershipgroup', '')
        self.noownershipgroup = self.module.params.get('noownershipgroup', '')
        self.removeallhosts = self.module.params.get('removeallhosts', '')
        if not self.name:
            self.module.fail_json(msg='Missing mandatory parameter: name')
        self.restapi = IBMSVCRestApi(module=self.module, clustername=self.module.params['clustername'], domain=self.module.params['domain'], username=self.module.params['username'], password=self.module.params['password'], validate_certs=self.module.params['validate_certs'], log_path=log_path, token=self.module.params['token'])

    def get_existing_hostcluster(self):
        merged_result = {}
        data = self.restapi.svc_obj_info(cmd='lshostcluster', cmdopts=None, cmdargs=[self.name])
        if isinstance(data, list):
            for d in data:
                merged_result.update(d)
        else:
            merged_result = data
        return merged_result

    def hostcluster_probe(self, data):
        props = []
        if self.removeallhosts:
            self.module.fail_json(msg="Parameter 'removeallhosts' can be used only while deleting hostcluster")
        if self.ownershipgroup and self.noownershipgroup:
            self.module.fail_json(msg="You must not pass in both 'ownershipgroup' and 'noownershipgroup' to the module.")
        if data['owner_name'] and self.noownershipgroup:
            props += ['noownershipgroup']
        if self.ownershipgroup and (not data['owner_name'] or self.ownershipgroup != data['owner_name']):
            props += ['ownershipgroup']
        if props is []:
            props = None
        self.log("hostcluster_probe props='%s'", data)
        return props

    def hostcluster_create(self):
        if self.removeallhosts:
            self.module.fail_json(msg="Parameter 'removeallhosts' cannot be passed while creating hostcluster")
        if self.module.check_mode:
            self.changed = True
            return
        cmd = 'mkhostcluster'
        cmdopts = {'name': self.name}
        if self.ownershipgroup:
            cmdopts['ownershipgroup'] = self.ownershipgroup
        self.log("creating host cluster command opts '%s'", self.ownershipgroup)
        result = self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
        self.log("create host cluster result '%s'", result)
        if 'message' in result:
            self.changed = True
            self.log("create host cluster result message '%s'", result['message'])
        else:
            self.module.fail_json(msg='Failed to create host cluster [%s]' % self.name)

    def hostcluster_update(self, modify):
        if self.module.check_mode:
            self.changed = True
            return
        self.log("updating host cluster '%s'", self.name)
        cmd = 'chhostcluster'
        cmdopts = {}
        if 'ownershipgroup' in modify:
            cmdopts['ownershipgroup'] = self.ownershipgroup
        elif 'noownershipgroup' in modify:
            cmdopts['noownershipgroup'] = self.noownershipgroup
        if cmdopts:
            cmdargs = [self.name]
            self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
            self.changed = True
            self.log('Properties of %s updated', self.name)

    def hostcluster_delete(self):
        if self.module.check_mode:
            self.changed = True
            return
        self.log("deleting host cluster '%s'", self.name)
        cmd = 'rmhostcluster'
        cmdopts = {}
        cmdargs = [self.name]
        if self.removeallhosts:
            cmdopts = {'force': True}
            cmdopts['removeallhosts'] = self.removeallhosts
        self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
        self.changed = True

    def apply(self):
        changed = False
        msg = None
        modify = []
        hc_data = self.get_existing_hostcluster()
        if hc_data:
            if self.state == 'absent':
                self.log("CHANGED: host cluster exists, but requested state is 'absent'")
                changed = True
            elif self.state == 'present':
                modify = self.hostcluster_probe(hc_data)
                if modify:
                    changed = True
        elif self.state == 'present':
            self.log("CHANGED: host cluster does not exist, but requested state is 'present'")
            changed = True
        if changed:
            if self.state == 'present':
                if not hc_data:
                    self.hostcluster_create()
                    msg = 'host cluster %s has been created.' % self.name
                else:
                    self.hostcluster_update(modify)
                    msg = 'host cluster [%s] has been modified.' % self.name
            elif self.state == 'absent':
                self.hostcluster_delete()
                msg = 'host cluster [%s] has been deleted.' % self.name
            if self.module.check_mode:
                msg = 'skipping changes due to check mode'
        else:
            self.log('exiting with no changes')
            if self.state == 'absent':
                msg = 'host cluster [%s] did not exist.' % self.name
            else:
                msg = 'host cluster [%s] already exists. No modifications done.' % self.name
        self.module.exit_json(msg=msg, changed=changed)