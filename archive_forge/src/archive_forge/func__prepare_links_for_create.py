from __future__ import absolute_import, division, print_function
import hashlib
import os
import re
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _prepare_links_for_create(self, params_dict):
    links_and_params = list()
    if self.want.key_name:
        key_link = 'https://{0}:{1}/mgmt/tm/sys/file/ssl-key/'.format(self.client.provider['server'], self.client.provider['server_port'])
        key_params_dict = params_dict.copy()
        key_params_dict['name'] = self.want.key_filename
        key_params_dict['sourcePath'] = self.want.key_source_path + '_key'
        if self.want.passphrase:
            key_params_dict['passphrase'] = self.want.passphrase
        links_and_params.append({'link': key_link, 'params': key_params_dict})
    if self.want.cert_name:
        cert_link = 'https://{0}:{1}/mgmt/tm/sys/file/ssl-cert/'.format(self.client.provider['server'], self.client.provider['server_port'])
        cert_params_dict = params_dict.copy()
        cert_params_dict['name'] = self.want.cert_filename
        cert_params_dict['sourcePath'] = self.want.cert_source_path + '_cert'
        links_and_params.append({'link': cert_link, 'params': cert_params_dict})
    return links_and_params