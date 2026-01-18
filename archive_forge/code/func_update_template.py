from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def update_template(self, template):
    args = {'id': template['id'], 'displaytext': self.get_or_fallback('display_text', 'name'), 'format': self.module.params.get('format'), 'isdynamicallyscalable': self.module.params.get('is_dynamically_scalable'), 'isrouting': self.module.params.get('is_routing'), 'ostypeid': self.get_os_type(key='id'), 'passwordenabled': self.module.params.get('password_enabled')}
    if self.has_changed(args, template):
        self.result['changed'] = True
        if not self.module.check_mode:
            self.query_api('updateTemplate', **args)
            template = self.get_template()
    args = {'id': template['id'], 'isextractable': self.module.params.get('is_extractable'), 'isfeatured': self.module.params.get('is_featured'), 'ispublic': self.module.params.get('is_public')}
    if self.has_changed(args, template):
        self.result['changed'] = True
        if not self.module.check_mode:
            self.query_api('updateTemplatePermissions', **args)
            template = self.get_template()
    if template:
        template = self.ensure_tags(resource=template, resource_type='Template')
    return template