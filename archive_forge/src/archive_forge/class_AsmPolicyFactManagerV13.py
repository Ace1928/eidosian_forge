from __future__ import absolute_import, division, print_function
import datetime
import math
import re
import time
import traceback
from collections import namedtuple
from ansible.module_utils.basic import (
from ansible.module_utils.parsing.convert_bool import BOOLEANS_TRUE
from ansible.module_utils.six import (
from ansible.module_utils.urls import urlparse
from ipaddress import ip_interface
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.urls import parseStats
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
class AsmPolicyFactManagerV13(AsmPolicyFactManager):

    def _exec_module(self):
        results = []
        facts = self.read_facts()
        for item in facts:
            attrs = item.to_return()
            results.append(attrs)
        results = sorted(results, key=lambda k: k['full_path'])
        return results

    def read_collection_from_device(self, skip=0):
        uri = 'https://{0}:{1}/mgmt/tm/asm/policies'.format(self.client.provider['server'], self.client.provider['server_port'])
        to_expand = 'general,signature-settings,header-settings,cookie-settings,antivirus,policy-builder,csrf-protection,csrf-urls'
        query = '?$top=10&$skip={0}&$expand={1}&$filter=partition+eq+{2}'.format(skip, to_expand, self.module.params['partition'])
        resp = self.client.api.get(uri + query)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status not in [200, 201] or ('code' in response and response['code'] not in [200, 201]):
            raise F5ModuleError(resp.content)
        if 'items' not in response:
            return []
        return response['items']