from __future__ import absolute_import, division, print_function
import abc
import os
import re
import shlex
from functools import partial
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.text.formatters import human_to_bytes
from ansible.module_utils.six import string_types
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._platform import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import (
def _preprocess_networks(module, values):
    if module.params['networks_cli_compatible'] is True and values.get('networks') and ('network_mode' not in values):
        values['network_mode'] = values['networks'][0]['name']
    if 'networks' in values:
        for network in values['networks']:
            if network['links']:
                parsed_links = []
                for link in network['links']:
                    parsed_link = link.split(':', 1)
                    if len(parsed_link) == 1:
                        parsed_link = (link, link)
                    parsed_links.append(tuple(parsed_link))
                network['links'] = parsed_links
            if network['mac_address']:
                network['mac_address'] = network['mac_address'].replace('-', ':')
    return values