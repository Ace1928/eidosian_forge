from __future__ import absolute_import, division, print_function
import json
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import missing_required_lib
from ..module_utils.gcp_utils import (
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
def _format_network_info(self, address):
    """
        :param address: A GCP network address
        :return a dict with network shortname and region
        """
    split = address.split('/')
    region = ''
    if 'global' in split:
        region = 'global'
    else:
        region = split[8]
    return {'region': region, 'name': split[-1], 'selfLink': address}