from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.frr.frr.plugins.module_utils.network.frr.providers.cli.config.bgp.address_family import (
from ansible_collections.frr.frr.plugins.module_utils.network.frr.providers.cli.config.bgp.neighbors import (
from ansible_collections.frr.frr.plugins.module_utils.network.frr.providers.providers import (
def _render_log_neighbor_changes(self, config=None):
    cmd = 'bgp log-neighbor-changes'
    log_neighbor_changes = self.get_value('config.log_neighbor_changes')
    if log_neighbor_changes is True:
        if not config or cmd not in config:
            return cmd
    elif log_neighbor_changes is False:
        if config and cmd in config:
            return 'no %s' % cmd