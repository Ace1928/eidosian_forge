from __future__ import absolute_import, division, print_function
import json
import re
import time
from ansible.errors import AnsibleConnectionFailure
from ansible.module_utils._text import to_text
from ansible.module_utils.common._collections_compat import Mapping
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.ansible.netcommon.plugins.plugin_utils.cliconf_base import (
def _extract_banners(self, config):
    banners = {}
    banner_cmds = re.findall('^banner (\\w+)', config, re.M)
    for cmd in banner_cmds:
        regex = 'banner %s \\^C(.+?)(?=\\^C)' % cmd
        match = re.search(regex, config, re.S)
        if match:
            key = 'banner %s' % cmd
            banners[key] = match.group(1).strip()
    for cmd in banner_cmds:
        regex = 'banner %s \\^C(.+?)(?=\\^C)' % cmd
        match = re.search(regex, config, re.S)
        if match:
            config = config.replace(str(match.group(1)), '')
    config = re.sub('banner \\w+ \\^C\\^C', '!! banner removed', config)
    return (config, banners)