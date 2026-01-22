from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.consul import (
class ConsulBindingRuleModule(_ConsulModule):
    api_endpoint = 'acl/binding-rule'
    result_key = 'binding_rule'
    unique_identifier = 'id'

    def read_object(self):
        url = 'acl/binding-rules?authmethod={0}'.format(self.params['auth_method'])
        try:
            results = self.get(url)
            for result in results:
                if result.get('Description').startswith('{0}: '.format(self.params['name'])):
                    return result
        except RequestError as e:
            if e.status == 404:
                return
            elif e.status == 403 and b'ACL not found' in e.response_data:
                return
            raise

    def module_to_obj(self, is_update):
        obj = super(ConsulBindingRuleModule, self).module_to_obj(is_update)
        del obj['Name']
        return obj

    def prepare_object(self, existing, obj):
        final = super(ConsulBindingRuleModule, self).prepare_object(existing, obj)
        name = self.params['name']
        description = final.pop('Description', '').split(': ', 1)[-1]
        final['Description'] = '{0}: {1}'.format(name, description)
        return final