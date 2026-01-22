from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.oneview import OneViewModuleBase
class SanManagerInfoModule(OneViewModuleBase):
    argument_spec = dict(provider_display_name=dict(type='str'), params=dict(type='dict'))

    def __init__(self):
        super(SanManagerInfoModule, self).__init__(additional_arg_spec=self.argument_spec, supports_check_mode=True)
        self.resource_client = self.oneview_client.san_managers

    def execute_module(self):
        if self.module.params.get('provider_display_name'):
            provider_display_name = self.module.params['provider_display_name']
            san_manager = self.oneview_client.san_managers.get_by_provider_display_name(provider_display_name)
            if san_manager:
                resources = [san_manager]
            else:
                resources = []
        else:
            resources = self.oneview_client.san_managers.get_all(**self.facts_params)
        return dict(changed=False, san_managers=resources)