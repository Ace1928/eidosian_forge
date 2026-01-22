from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.vultr_v2 import AnsibleVultr, vultr_argument_spec
class AnsibleVultrBlockStorage(AnsibleVultr):

    def update(self, resource):
        current_size = resource['size_gb']
        desired_size = self.module.params['size_gb']
        if desired_size < current_size:
            self.module.params['size_gb'] = current_size
            self.module.warn('Shrinking is not supported: current size %s, desired size %s' % (current_size, desired_size))
        return super(AnsibleVultrBlockStorage, self).update(resource=resource)

    def present(self):
        resource = self.create_or_update() or dict()
        instance_to_attach = self.module.params.get('attached_to_instance')
        if instance_to_attach is None:
            self.get_result(resource)
        instance_attached = resource.get('attached_to_instance', '')
        if instance_attached != instance_to_attach:
            self.result['changed'] = True
            mode = 'detach' if not instance_to_attach else 'attach'
            self.result['diff']['after'].update({'attached_to_instance': instance_to_attach})
            data = {'instance_id': instance_to_attach if instance_to_attach else None, 'live': self.module.params.get('live')}
            if not self.module.check_mode:
                self.api_query(path='%s/%s/%s' % (self.resource_path, resource[self.resource_key_id], mode), method='POST', data=data)
                resource = self.query_by_id(resource_id=resource[self.resource_key_id])
        self.get_result(resource)