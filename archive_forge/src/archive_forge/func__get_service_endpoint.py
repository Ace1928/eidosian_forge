from __future__ import (absolute_import, division, print_function)
import re
import time
import traceback
from ansible.module_utils.basic import (AnsibleModule, env_fallback,
from ansible.module_utils.common.text.converters import to_text
def _get_service_endpoint(self, client, service_type, region):
    k = '%s.%s' % (service_type, region if region else '')
    if k in self._endpoints:
        return self._endpoints.get(k)
    url = None
    try:
        url = client.get_endpoint(service_type=service_type, region_name=region, interface='public')
    except Exception as ex:
        raise HwcClientException(0, 'Getting endpoint failed, error=%s' % ex)
    if url == '':
        raise HwcClientException(0, 'Cannot find the endpoint for %s' % service_type)
    if url[-1] != '/':
        url += '/'
    self._endpoints[k] = url
    return url