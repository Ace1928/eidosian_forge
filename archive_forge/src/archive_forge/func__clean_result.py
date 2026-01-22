from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
def _clean_result(self, result):
    """
        Will clean the result from irrelevant fields
        :param result: The result from the query
        :return: The modified result
        """
    del result['utm_host']
    del result['utm_port']
    del result['utm_token']
    del result['utm_protocol']
    del result['validate_certs']
    del result['url_username']
    del result['url_password']
    del result['state']
    return result