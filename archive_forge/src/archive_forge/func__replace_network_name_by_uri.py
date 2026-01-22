from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.oneview import OneViewModuleBase, OneViewModuleResourceNotFound
def _replace_network_name_by_uri(self, data):
    if 'networkUris' in data:
        data['networkUris'] = [self._get_network_uri(x) for x in data['networkUris']]