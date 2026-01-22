from __future__ import absolute_import, division, print_function
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _get_server_tech_link(self):
    uri = 'https://{0}:{1}/mgmt/tm/asm/server-technologies/'.format(self.client.provider['server'], self.client.provider['server_port'])
    name = self.want.name.replace(' ', '%20')
    query = "?$filter=contains(serverTechnologyName,'{0}')".format(name)
    resp = self.client.api.get(uri + query)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if resp.status not in [200, 201] or ('code' in response and response['code'] not in [200, 201]):
        raise F5ModuleError(resp.content)
    if 'items' in response and response['items'] != []:
        for item in response['items']:
            if item['serverTechnologyName'] == self.want.name:
                return item['selfLink']
    raise F5ModuleError('The following server technology: {0} was not found on the device.'.format(self.want.name))