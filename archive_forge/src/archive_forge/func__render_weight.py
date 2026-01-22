from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.arista.eos.plugins.module_utils.network.eos.providers.providers import (
def _render_weight(self, item, config=None):
    cmd = 'neighbor %s weight %s' % (item['neighbor'], item['weight'])
    if not config or cmd not in config:
        return cmd