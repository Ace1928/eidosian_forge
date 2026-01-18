from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def update_instance_license_type(self, api_root, rest_api, headers, instance_type, license_type):
    body = {'instanceType': instance_type, 'licenseType': license_type}
    response, err, dummy = rest_api.put(api_root + 'license-instance-type', body, header=headers)
    if err is not None:
        return (False, 'Error: unexpected response on modify instance_type and license_type: %s, %s' % (str(err), str(response)))
    dummy, err = self.wait_cvo_update_complete(rest_api, headers)
    return (err is None, err)