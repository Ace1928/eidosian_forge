from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
class AnsibleCloudStackTemplate(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackTemplate, self).__init__(module)
        self.returns = {'checksum': 'checksum', 'status': 'status', 'isready': 'is_ready', 'templatetag': 'template_tag', 'sshkeyenabled': 'sshkey_enabled', 'passwordenabled': 'password_enabled', 'templatetype': 'template_type', 'ostypename': 'os_type', 'crossZones': 'cross_zones', 'format': 'format', 'hypervisor': 'hypervisor', 'url': 'url', 'extractMode': 'mode', 'state': 'state'}

    def _get_args(self):
        args = {'name': self.module.params.get('name'), 'displaytext': self.get_or_fallback('display_text', 'name'), 'bits': self.module.params.get('bits'), 'isdynamicallyscalable': self.module.params.get('is_dynamically_scalable'), 'isextractable': self.module.params.get('is_extractable'), 'isfeatured': self.module.params.get('is_featured'), 'ispublic': self.module.params.get('is_public'), 'passwordenabled': self.module.params.get('password_enabled'), 'requireshvm': self.module.params.get('requires_hvm'), 'templatetag': self.module.params.get('template_tag'), 'ostypeid': self.get_os_type(key='id')}
        if not args['ostypeid']:
            self.module.fail_json(msg='Missing required arguments: os_type')
        return args

    def get_root_volume(self, key=None):
        args = {'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id'), 'virtualmachineid': self.get_vm(key='id'), 'type': 'ROOT'}
        volumes = self.query_api('listVolumes', **args)
        if volumes:
            return self._get_by_key(key, volumes['volume'][0])
        self.module.fail_json(msg="Root volume for '%s' not found" % self.get_vm('name'))

    def get_snapshot(self, key=None):
        snapshot = self.module.params.get('snapshot')
        if not snapshot:
            return None
        args = {'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id'), 'volumeid': self.get_root_volume('id'), 'fetch_list': True}
        snapshots = self.query_api('listSnapshots', **args)
        if snapshots:
            for s in snapshots:
                if snapshot in [s['name'], s['id']]:
                    return self._get_by_key(key, s)
        self.module.fail_json(msg="Snapshot '%s' not found" % snapshot)

    def present_template(self):
        template = self.get_template()
        if template:
            template = self.update_template(template)
        elif self.module.params.get('url'):
            template = self.register_template()
        elif self.module.params.get('vm'):
            template = self.create_template()
        else:
            self.fail_json(msg='one of the following is required on state=present: url, vm')
        return template

    def create_template(self):
        template = None
        self.result['changed'] = True
        args = self._get_args()
        snapshot_id = self.get_snapshot(key='id')
        if snapshot_id:
            args['snapshotid'] = snapshot_id
        else:
            args['volumeid'] = self.get_root_volume('id')
        if not self.module.check_mode:
            template = self.query_api('createTemplate', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                template = self.poll_job(template, 'template')
        if template:
            template = self.ensure_tags(resource=template, resource_type='Template')
        return template

    def register_template(self):
        required_params = ['format', 'url', 'hypervisor']
        self.module.fail_on_missing_params(required_params=required_params)
        template = None
        self.result['changed'] = True
        args = self._get_args()
        args.update({'url': self.module.params.get('url'), 'format': self.module.params.get('format'), 'checksum': self.module.params.get('checksum'), 'isextractable': self.module.params.get('is_extractable'), 'isrouting': self.module.params.get('is_routing'), 'sshkeyenabled': self.module.params.get('sshkey_enabled'), 'hypervisor': self.get_hypervisor(), 'domainid': self.get_domain(key='id'), 'account': self.get_account(key='name'), 'projectid': self.get_project(key='id')})
        if not self.module.params.get('cross_zones'):
            args['zoneid'] = self.get_zone(key='id')
        else:
            args['zoneid'] = -1
        if not self.module.check_mode:
            self.query_api('registerTemplate', **args)
            template = self.get_template()
        return template

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

    def _is_find_option(self, param_name):
        return param_name in self.module.params.get('template_find_options')

    def _find_option_match(self, template, param_name, internal_name=None):
        if not internal_name:
            internal_name = param_name
        if param_name in self.module.params.get('template_find_options'):
            param_value = self.module.params.get(param_name)
            if not param_value:
                self.fail_json(msg='The param template_find_options has %s but param was not provided.' % param_name)
            if template[internal_name] == param_value:
                return True
        return False

    def get_template(self):
        args = {'name': self.module.params.get('name'), 'templatefilter': self.module.params.get('template_filter'), 'domainid': self.get_domain(key='id'), 'account': self.get_account(key='name'), 'projectid': self.get_project(key='id')}
        cross_zones = self.module.params.get('cross_zones')
        if not cross_zones:
            args['zoneid'] = self.get_zone(key='id')
        template_found = None
        templates = self.query_api('listTemplates', **args)
        if templates:
            for tmpl in templates['template']:
                if self._is_find_option('cross_zones') and (not self._find_option_match(template=tmpl, param_name='cross_zones', internal_name='crossZones')):
                    continue
                if self._is_find_option('checksum') and (not self._find_option_match(template=tmpl, param_name='checksum')):
                    continue
                if self._is_find_option('display_text') and (not self._find_option_match(template=tmpl, param_name='display_text', internal_name='displaytext')):
                    continue
                if not template_found:
                    template_found = tmpl
                elif tmpl['id'] == template_found['id']:
                    continue
                else:
                    self.fail_json(msg='Multiple templates found matching provided params. Please use template_find_options.')
        return template_found

    def extract_template(self):
        template = self.get_template()
        if not template:
            self.module.fail_json(msg='Failed: template not found')
        if self.module.params.get('cross_zones'):
            self.module.warn('cross_zones parameter is ignored when state is extracted')
        args = {'id': template['id'], 'url': self.module.params.get('url'), 'mode': self.module.params.get('mode'), 'zoneid': self.get_zone(key='id')}
        self.result['changed'] = True
        if not self.module.check_mode:
            template = self.query_api('extractTemplate', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                template = self.poll_job(template, 'template')
        return template

    def remove_template(self):
        template = self.get_template()
        if template:
            self.result['changed'] = True
            args = {'id': template['id']}
            if not self.module.params.get('cross_zones'):
                args['zoneid'] = self.get_zone(key='id')
            if not self.module.check_mode:
                res = self.query_api('deleteTemplate', **args)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    res = self.poll_job(res, 'template')
        return template

    def get_result(self, resource):
        super(AnsibleCloudStackTemplate, self).get_result(resource)
        if resource:
            if 'isextractable' in resource:
                self.result['is_extractable'] = True if resource['isextractable'] else False
            if 'isfeatured' in resource:
                self.result['is_featured'] = True if resource['isfeatured'] else False
            if 'ispublic' in resource:
                self.result['is_public'] = True if resource['ispublic'] else False
        return self.result