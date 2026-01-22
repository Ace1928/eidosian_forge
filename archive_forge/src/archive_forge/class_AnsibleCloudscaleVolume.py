from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.api import (
class AnsibleCloudscaleVolume(AnsibleCloudscaleBase):

    def create(self, resource):
        self._module.fail_on_missing_params(['name', 'size_gb'])
        return super(AnsibleCloudscaleVolume, self).create(resource)

    def find_difference(self, key, resource, param):
        is_different = False
        if key != 'servers':
            return super(AnsibleCloudscaleVolume, self).find_difference(key, resource, param)
        server_has = resource[key]
        server_wanted = param
        if len(server_wanted) != len(server_has):
            is_different = True
        else:
            for has in server_has:
                if has['uuid'] not in server_wanted:
                    is_different = True
        return is_different