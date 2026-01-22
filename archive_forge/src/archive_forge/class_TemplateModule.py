from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.opennebula import OpenNebulaModule
class TemplateModule(OpenNebulaModule):

    def __init__(self):
        argument_spec = dict(id=dict(type='int', required=False), name=dict(type='str', required=False), state=dict(type='str', choices=['present', 'absent'], default='present'), template=dict(type='str', required=False))
        mutually_exclusive = [['id', 'name']]
        required_one_of = [('id', 'name')]
        required_if = [['state', 'present', ['template']]]
        OpenNebulaModule.__init__(self, argument_spec, supports_check_mode=True, mutually_exclusive=mutually_exclusive, required_one_of=required_one_of, required_if=required_if)

    def run(self, one, module, result):
        params = module.params
        id = params.get('id')
        name = params.get('name')
        desired_state = params.get('state')
        template_data = params.get('template')
        self.result = {}
        template = self.get_template_instance(id, name)
        needs_creation = False
        if not template and desired_state != 'absent':
            if id:
                module.fail_json(msg='There is no template with id=' + str(id))
            else:
                needs_creation = True
        if desired_state == 'absent':
            self.result = self.delete_template(template)
        elif needs_creation:
            self.result = self.create_template(name, template_data)
        else:
            self.result = self.update_template(template, template_data)
        self.exit()

    def get_template(self, predicate):
        pool = self.one.templatepool.info(-3, -1, -1)
        for template in pool.VMTEMPLATE:
            if predicate(template):
                return template
        return None

    def get_template_by_id(self, template_id):
        return self.get_template(lambda template: template.ID == template_id)

    def get_template_by_name(self, name):
        return self.get_template(lambda template: template.NAME == name)

    def get_template_instance(self, requested_id, requested_name):
        if requested_id:
            return self.get_template_by_id(requested_id)
        else:
            return self.get_template_by_name(requested_name)

    def get_template_info(self, template):
        info = {'id': template.ID, 'name': template.NAME, 'template': template.TEMPLATE, 'user_name': template.UNAME, 'user_id': template.UID, 'group_name': template.GNAME, 'group_id': template.GID}
        return info

    def create_template(self, name, template_data):
        if not self.module.check_mode:
            self.one.template.allocate('NAME = "' + name + '"\n' + template_data)
        result = self.get_template_info(self.get_template_by_name(name))
        result['changed'] = True
        return result

    def update_template(self, template, template_data):
        if not self.module.check_mode:
            self.one.template.update(template.ID, template_data, 0)
        result = self.get_template_info(self.get_template_by_id(template.ID))
        if self.module.check_mode:
            result['changed'] = True
        else:
            result['changed'] = template.TEMPLATE != result['template']
        return result

    def delete_template(self, template):
        if not template:
            return {'changed': False}
        if not self.module.check_mode:
            self.one.template.delete(template.ID)
        return {'changed': True}