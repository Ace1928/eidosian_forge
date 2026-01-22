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
@property
def api_spec(self):
    if self._api_spec is None:
        spec_path_url = self._get_api_spec_path()
        response = self.send_request(url_path=spec_path_url, http_method=HTTPMethod.GET)
        if response[ResponseParams.SUCCESS]:
            self._api_spec = FdmSwaggerParser().parse_spec(response[ResponseParams.RESPONSE])
        else:
            raise ConnectionError('Failed to download API specification. Status code: %s. Response: %s' % (response[ResponseParams.STATUS_CODE], response[ResponseParams.RESPONSE]))
    return self._api_spec