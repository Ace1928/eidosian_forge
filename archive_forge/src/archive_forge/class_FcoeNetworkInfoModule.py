from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.oneview import OneViewModuleBase
class FcoeNetworkInfoModule(OneViewModuleBase):

    def __init__(self):
        argument_spec = dict(name=dict(type='str'), params=dict(type='dict'))
        super(FcoeNetworkInfoModule, self).__init__(additional_arg_spec=argument_spec, supports_check_mode=True)

    def execute_module(self):
        if self.module.params['name']:
            fcoe_networks = self.oneview_client.fcoe_networks.get_by('name', self.module.params['name'])
        else:
            fcoe_networks = self.oneview_client.fcoe_networks.get_all(**self.facts_params)
        return dict(changed=False, fcoe_networks=fcoe_networks)