from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import bigiq_version
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
class UsableChanges(Changes):

    @property
    def resources(self):
        result = dict()
        result.update(self.http_profile)
        result.update(self.http_monitor)
        result.update(self.virtual)
        result.update(self.pool)
        result.update(self.nodes)
        return result

    @property
    def virtual(self):
        result = dict()
        result['ltm:virtual::b487671f29ba'] = [dict(parameters=dict(name='virtual', destinationAddress=self.inbound_virtual['address'], mask=self.inbound_virtual['netmask'], destinationPort=self.inbound_virtual.get('port', 80)), subcollectionResources=self.profiles)]
        return result

    @property
    def profiles(self):
        result = {'profiles:9448fe71611e': [dict(parameters=dict())], 'profiles:03a4950ab656': [dict(parameters=dict())]}
        return result

    @property
    def pool(self):
        result = dict()
        result['ltm:pool:9a593d17495b'] = [dict(parameters=dict(name='pool_0'), subcollectionResources=self.pool_members)]
        return result

    @property
    def pool_members(self):
        result = dict()
        result['members:5109c66dfbac'] = []
        for x in self.servers:
            member = dict(parameters=dict(port=x.get('port', 80), nodeReference=dict(link='#/resources/ltm:node:9e76a6323321/{0}'.format(x['address']), fullPath='# {0}'.format(x['address']))))
            result['members:5109c66dfbac'].append(member)
        return result

    @property
    def http_profile(self):
        result = dict()
        result['ltm:profile:http:03a4950ab656'] = [dict(parameters=dict(name='profile_http'))]
        return result

    @property
    def http_monitor(self):
        result = dict()
        result['ltm:monitor:http:ea4346e49cdf'] = [dict(parameters=dict(name='monitor-http'))]
        return result

    @property
    def nodes(self):
        result = dict()
        result['ltm:node:9e76a6323321'] = []
        for x in self.servers:
            tmp = dict(parameters=dict(name=x['address'], address=x['address']))
            result['ltm:node:9e76a6323321'].append(tmp)
        return result

    @property
    def node_addresses(self):
        result = [x['address'] for x in self.servers]
        return result