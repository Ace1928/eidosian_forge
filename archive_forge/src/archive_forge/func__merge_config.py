from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.lxd import LXDClient, LXDClientException
from ansible.module_utils.six.moves.urllib.parse import urlencode
def _merge_config(self, config):
    """ merge profile

        Merge Configuration of the present profile and the new desired configitems

        Args:
            dict(config): Dict with the old config in 'metadata' and new config in 'config'
        Kwargs:
            None
        Raises:
            None
        Returns:
            dict(config): new config"""
    for item in ['config', 'description', 'devices', 'name', 'used_by']:
        if item in config:
            config[item] = self._merge_dicts(config['metadata'][item], config[item])
        else:
            config[item] = config['metadata'][item]
    return self._merge_dicts(self.config, config)