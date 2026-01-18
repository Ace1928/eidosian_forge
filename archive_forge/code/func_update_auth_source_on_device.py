from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def update_auth_source_on_device(self, source):
    """Set the system auth source.

        Configuring the authentication source is only one step in the process of setting
        up an auth source. The other step is to inform the system of the auth source
        you want to use.

        This method is used for situations where

        * The ``use_for_auth`` parameter is set to ``yes``
        * The ``use_for_auth`` parameter is set to ``no``
        * The ``state`` parameter is set to ``absent``

        When ``state`` equal to ``absent``, before you can delete the TACACS+ configuration,
        you must set the system auth to "something else". The system ships with a system
        auth called "local", so this is the logical "something else" to use.

        When ``use_for_auth`` is no, the same situation applies as when ``state`` equal
        to ``absent`` is done above.

        When ``use_for_auth`` is ``yes``, this method will set the current system auth
        state to TACACS+.

        Arguments:
            source (string): The source that you want to set on the device.
        """
    params = dict(type=source)
    uri = 'https://{0}:{1}/mgmt/tm/auth/source/'.format(self.client.provider['server'], self.client.provider['server_port'])
    resp = self.client.api.patch(uri, json=params)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if 'code' in response and response['code'] == 400:
        if 'message' in response:
            raise F5ModuleError(response['message'])
        else:
            raise F5ModuleError(resp.content)