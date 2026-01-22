from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _wait_for_cgnat_ready(self):
    """Waits specifically for CGNAT

        Starting in TMOS 15.0 cgnat can take longer to actually start up than all the previous checks take.
        This check here is specifically waiting for a cgnat API to stop raising
        errors.
        :return:
        """
    nops = 0
    while nops < 3:
        try:
            uri = 'https://{0}:{1}/mgmt/tm/ltm/lsn-pool'.format(self.client.provider['server'], self.client.provider['server_port'])
            resp = self.client.api.get(uri)
            try:
                response = resp.json()
            except ValueError as ex:
                raise F5ModuleError(str(ex))
            if 'code' in response and response['code'] in [400, 403]:
                if 'message' in response:
                    raise F5ModuleError(response['message'])
                else:
                    raise F5ModuleError(resp.content)
            if len(response['items']) >= 0:
                nops += 1
            else:
                nops = 0
        except Exception:
            pass
        time.sleep(5)