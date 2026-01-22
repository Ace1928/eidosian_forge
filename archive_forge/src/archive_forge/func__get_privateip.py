from __future__ import absolute_import, division, print_function
import json
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import missing_required_lib
from ..module_utils.gcp_utils import (
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
def _get_privateip(self):
    """
        :param item: A host response from GCP
        :return the privateIP of this instance or None
        """
    for interface in self.json['networkInterfaces']:
        if 'networkIP' in interface:
            return interface['networkIP']