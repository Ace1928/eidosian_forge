from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.frr.frr.plugins.module_utils.network.frr.providers.providers import (
def _render_remove_private_as(self, item, config=None):
    cmd = 'neighbor %s remove-private-AS' % item['neighbor']
    if item['remove_private_as'] is False:
        if not config or cmd in config:
            cmd = 'no %s' % cmd
            return cmd
    elif not config or cmd not in config:
        return cmd