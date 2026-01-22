from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
class ProtocolSipManager(BaseManager):
    """Manages AFM DoS Profile Protocol SIP settings.

    Protocol SIP settings are a sub-collection attached to each profile.

    There are many similar vectors that can be managed here. This configuration
    is a sub-set of the device-config DoS vector configuration and excludes
    several attributes per-vector that are found in the device-config configuration.
    These include,

      * rateIncrease
      * rateLimit
      * rateThreshold
    """

    def __init__(self, *args, **kwargs):
        super(ProtocolSipManager, self).__init__(**kwargs)
        self.want = ModuleParameters(params=self.module.params)
        self.have = ApiParameters()
        self.changes = UsableChanges()

    def update(self):
        return self._update('sipAttackVector')

    def update_on_device(self):
        params = self.changes.api_params()
        uri = 'https://{0}:{1}/mgmt/tm/security/dos/profile/{2}/protocol-sip/{3}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.profile), self.want.profile)
        resp = self.client.api.patch(uri, json=params)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
            return True
        raise F5ModuleError(resp.content)

    def read_current_from_device(self):
        uri = 'https://{0}:{1}/mgmt/tm/security/dos/profile/{2}/protocol-sip/{3}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.profile), self.want.profile)
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status == 404 or ('code' in response and response['code'] == 404):
            self.create_vector_container_on_device()
            return self.read_current_from_device()
        if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
            return response.get('sipAttackVector', [])
        raise F5ModuleError(resp.content)

    def create_vector_container_on_device(self):
        params = {'name': self.want.profile}
        uri = 'https://{0}:{1}/mgmt/tm/security/dos/profile/{2}/protocol-sip/'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.profile))
        resp = self.client.api.post(uri, json=params)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
            return True
        raise F5ModuleError(resp.content)