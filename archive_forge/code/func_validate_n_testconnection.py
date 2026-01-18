from __future__ import (absolute_import, division, print_function)
import json
import os
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.common.dict_transformations import recursive_diff
def validate_n_testconnection(module, rest_obj, payload):
    dc_cnt = {'DNS': 1, 'MANUAL': 3}
    dc_type = {'DNS': 'DnsServer', 'MANUAL': 'ServerName'}
    dc_lookup = payload.get('ServerType')
    if len(payload.get(dc_type[dc_lookup])) > dc_cnt[dc_lookup]:
        module.fail_json(msg=DOMAIN_ALLOWED_COUNT.format(dc_lookup, dc_cnt[dc_lookup]))
    t_list = ['NetworkTimeOut', 'SearchTimeOut']
    for tx in t_list:
        if payload.get(tx) not in range(MIN_TIMEOUT, MAX_TIMEOUT + 1):
            module.fail_json(msg=TIMEOUT_RANGE.format(tx, MIN_TIMEOUT, MAX_TIMEOUT))
    payload['CertificateFile'] = ''
    if payload.get('CertificateValidation'):
        cert_path = module.params.get('certificate_file')
        if os.path.exists(cert_path):
            with open(cert_path, 'r') as certfile:
                cert_data = certfile.read()
                payload['CertificateFile'] = cert_data
        else:
            module.fail_json(msg=CERT_INVALID)
    msg = ''
    if module.params.get('test_connection'):
        test_connection(module, rest_obj, payload)
        msg = TEST_CONNECTION_SUCCESS
    return msg