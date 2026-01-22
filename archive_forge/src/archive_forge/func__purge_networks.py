from __future__ import absolute_import, division, print_function
import re
from time import sleep
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import parse_repository_tag
def _purge_networks(self, container, networks):
    for network in networks:
        self.results['actions'].append(dict(removed_from_network=network['name']))
        if not self.check_mode:
            try:
                self.engine_driver.disconnect_container_from_network(self.client, container.id, network['name'])
            except Exception as exc:
                self.fail('Error disconnecting container from network %s - %s' % (network['name'], to_native(exc)))
    return self._get_container(container.id)