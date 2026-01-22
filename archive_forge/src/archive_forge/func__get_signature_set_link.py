from __future__ import absolute_import, division, print_function
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _get_signature_set_link(self):
    result = None
    signature_set = self.want.name
    uri = 'https://{0}:{1}/mgmt/tm/asm/signature-sets'.format(self.client.provider['server'], self.client.provider['server_port'])
    query = '?$select=name'
    resp = self.client.api.get(uri + query)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if resp.status not in [200, 201] or ('code' in response and response['code'] not in [200, 201]):
        raise F5ModuleError(resp.content)
    if 'items' in response and response['items'] != []:
        for item in response['items']:
            if item['name'] == signature_set:
                result = dict(link=item['selfLink'])
    if result:
        return result
    raise F5ModuleError('The following signature set: {0} was not found on the device. Possibly name has changed in your TMOS version.'.format(self.want.name))