from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import string_types
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import bigiq_version
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
def _get_cert_references(self):
    result = dict()
    uri = 'https://{0}:{1}/mgmt/cm/adc-core/working-config/sys/file/ssl-cert/'.format(self.client.provider['server'], self.client.provider['server_port'])
    resp = self.client.api.get(uri)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    for cert in response['items']:
        key = fq_name(cert['partition'], cert['name'])
        result[key] = cert['selfLink']
    return result