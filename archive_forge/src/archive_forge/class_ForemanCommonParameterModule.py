from __future__ import absolute_import, division, print_function
from ansible_collections.theforeman.foreman.plugins.module_utils.foreman_helper import ForemanEntityAnsibleModule, parameter_value_to_str
class ForemanCommonParameterModule(ForemanEntityAnsibleModule):

    def remove_sensitive_fields(self, entity):
        if entity and 'hidden_value?' in entity:
            entity['hidden_value'] = entity.pop('hidden_value?')
            if entity['hidden_value']:
                entity['value'] = None
        return super(ForemanCommonParameterModule, self).remove_sensitive_fields(entity)