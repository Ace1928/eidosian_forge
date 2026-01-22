from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.scaleway import (
class ScalewaySecurityGroupInfo(Scaleway):

    def __init__(self, module):
        super(ScalewaySecurityGroupInfo, self).__init__(module)
        self.name = 'security_groups'
        region = module.params['region']
        self.module.params['api_url'] = SCALEWAY_LOCATION[region]['api_endpoint']