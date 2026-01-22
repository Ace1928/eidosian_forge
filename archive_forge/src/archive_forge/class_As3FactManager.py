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
class As3FactManager(BaseManager):

    def __init__(self, *args, **kwargs):
        self.client = kwargs.get('client', None)
        self.module = kwargs.get('module', None)
        self.installed_packages = packages_installed(self.client)
        super(As3FactManager, self).__init__(**kwargs)

    def exec_module(self):
        facts = self._exec_module()
        result = dict(as3_config=facts)
        return result

    def _exec_module(self):
        if 'as3' not in self.installed_packages:
            return []
        facts = self.read_facts()
        return facts

    def read_facts(self):
        collection = self.read_collection_from_device()
        return collection

    def read_collection_from_device(self):
        uri = 'https://{0}:{1}/mgmt/shared/appsvcs/declare'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status == 204 or ('code' in response and response['code'] == 204):
            return []
        if resp.status not in [200, 201] or ('code' in response and response['code'] not in [200, 201]):
            raise F5ModuleError(resp.content)
        if 'class' not in response:
            return []
        result = dict()
        result['declaration'] = response
        return result