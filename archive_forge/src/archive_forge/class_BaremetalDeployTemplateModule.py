from ansible_collections.openstack.cloud.plugins.module_utils.openstack import (
class BaremetalDeployTemplateModule(OpenStackModule):
    argument_spec = dict(extra=dict(type='dict'), id=dict(aliases=['uuid']), name=dict(), steps=dict(type='list', elements='dict'), state=dict(default='present', choices=['present', 'absent']))
    module_kwargs = dict(required_one_of=[('id', 'name')])

    def run(self):
        template = self._find_deploy_template()
        state = self.params['state']
        if state == 'present':
            kwargs = {}
            for k in ['extra', 'id', 'name', 'steps']:
                if self.params[k] is not None:
                    kwargs[k] = self.params[k]
            changed = True
            if not template:
                template = self.conn.baremetal.create_deploy_template(**kwargs)
            else:
                updates = dict(((k, v) for k, v in kwargs.items() if v != template[k]))
                if updates:
                    template = self.conn.baremetal.update_deploy_template(template['id'], **updates)
                else:
                    changed = False
            self.exit_json(changed=changed, template=template.to_dict(computed=False))
        if state == 'absent':
            if not template:
                self.exit_json(changed=False)
            template = self.conn.baremetal.delete_deploy_template(template['id'])
            self.exit_json(changed=True)

    def _find_deploy_template(self):
        id_or_name = self.params['id'] if self.params['id'] else self.params['name']
        try:
            return self.conn.baremetal.get_deploy_template(id_or_name)
        except self.sdk.exceptions.ResourceNotFound:
            return None