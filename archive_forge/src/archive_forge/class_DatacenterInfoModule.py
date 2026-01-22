from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.oneview import OneViewModuleBase
class DatacenterInfoModule(OneViewModuleBase):
    argument_spec = dict(name=dict(type='str'), options=dict(type='list', elements='str'), params=dict(type='dict'))

    def __init__(self):
        super(DatacenterInfoModule, self).__init__(additional_arg_spec=self.argument_spec, supports_check_mode=True)

    def execute_module(self):
        client = self.oneview_client.datacenters
        info = {}
        if self.module.params.get('name'):
            datacenters = client.get_by('name', self.module.params['name'])
            if self.options and 'visualContent' in self.options:
                if datacenters:
                    info['datacenter_visual_content'] = client.get_visual_content(datacenters[0]['uri'])
                else:
                    info['datacenter_visual_content'] = None
            info['datacenters'] = datacenters
        else:
            info['datacenters'] = client.get_all(**self.facts_params)
        return dict(changed=False, **info)