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
class OneConnectProfilesFactManager(BaseManager):

    def __init__(self, *args, **kwargs):
        self.client = kwargs.get('client', None)
        self.module = kwargs.get('module', None)
        super(OneConnectProfilesFactManager, self).__init__(**kwargs)

    def exec_module(self):
        facts = self._exec_module()
        result = dict(oneconnect_profiles=facts)
        return result

    def _exec_module(self):
        results = []
        facts = self.read_facts()
        for item in facts:
            attrs = item.to_return()
            results.append(attrs)
        results = sorted(results, key=lambda k: k['full_path'])
        return results

    def read_facts(self):
        results = []
        collection = self.increment_read()
        for resource in collection:
            params = OneConnectProfilesParameters(params=resource)
            results.append(params)
        return results

    def increment_read(self):
        n = 0
        result = []
        while True:
            items = self.read_collection_from_device(skip=n)
            if not items:
                break
            result.extend(items)
            n = n + self.module.params['data_increment']
        return result

    def read_collection_from_device(self, skip=0):
        uri = 'https://{0}:{1}/mgmt/tm/ltm/profile/one-connect'.format(self.client.provider['server'], self.client.provider['server_port'])
        query = '?$top={0}&$skip={1}&$filter=partition+eq+{2}'.format(self.module.params['data_increment'], skip, self.module.params['partition'])
        resp = self.client.api.get(uri + query)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status not in [200, 201] or ('code' in response and response['code'] not in [200, 201]):
            raise F5ModuleError(resp.content)
        if 'items' not in response:
            return []
        result = response['items']
        return result