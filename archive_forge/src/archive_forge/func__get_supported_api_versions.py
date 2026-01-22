from __future__ import (absolute_import, division, print_function)
import json
import os
import re
from ansible import __version__ as ansible_version
from ansible.module_utils.basic import to_text
from ansible.errors import AnsibleConnectionFailure, AnsibleError
from ansible_collections.community.network.plugins.module_utils.network.ftd.fdm_swagger_client import FdmSwaggerParser, SpecProp, FdmSwaggerValidator
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod, ResponseParams
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.plugins.httpapi import HttpApiBase
from ansible.module_utils.connection import ConnectionError
def _get_supported_api_versions(self):
    """
        Fetch list of API versions supported by device.

        :return: list of API versions suitable for device
        :rtype: list
        """
    http_method = HTTPMethod.GET
    response, response_data = self._send_service_request(path=GET_API_VERSIONS_PATH, error_msg_prefix="Can't fetch list of supported api versions", method=http_method, headers=BASE_HEADERS)
    value = self._get_response_value(response_data)
    self._display(http_method, 'response', value)
    api_versions_info = self._response_to_json(value)
    return api_versions_info['supportedVersions']