from __future__ import absolute_import, division, print_function
import re
from time import sleep
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import parse_repository_tag
def _add_networks(self, container, differences):
    for diff in differences:
        if diff.get('container'):
            self.results['actions'].append(dict(removed_from_network=diff['parameter']['name']))
            if not self.check_mode:
                try:
                    self.engine_driver.disconnect_container_from_network(self.client, container.id, diff['parameter']['id'])
                except Exception as exc:
                    self.fail('Error disconnecting container from network %s - %s' % (diff['parameter']['name'], to_native(exc)))
        self.results['actions'].append(dict(added_to_network=diff['parameter']['name'], network_parameters=diff['parameter']))
        if not self.check_mode:
            params = {key: value for key, value in diff['parameter'].items() if key not in ('id', 'name')}
            try:
                self.log('Connecting container to network %s' % diff['parameter']['id'])
                self.log(params, pretty_print=True)
                self.engine_driver.connect_container_to_network(self.client, container.id, diff['parameter']['id'], params)
            except Exception as exc:
                self.fail('Error connecting container to network %s - %s' % (diff['parameter']['name'], to_native(exc)))
    return self._get_container(container.id)