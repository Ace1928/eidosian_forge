from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def update_svm_password(self, api_root, rest_api, headers, svm_password):
    body = {'password': svm_password}
    response, err, dummy = rest_api.put(api_root + 'set-password', body, header=headers)
    if err is not None:
        return (False, 'Error: unexpected response on modifying svm_password: %s, %s' % (str(err), str(response)))
    return (True, None)