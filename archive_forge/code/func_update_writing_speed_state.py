from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def update_writing_speed_state(self, api_root, rest_api, headers, writing_speed_state):
    body = {'writingSpeedState': writing_speed_state.upper()}
    response, err, dummy = rest_api.put(api_root + 'writing-speed', body, header=headers)
    if err is not None:
        return (False, 'Error: unexpected response on modify writing_speed_state: %s, %s' % (str(err), str(response)))
    dummy, err = self.wait_cvo_update_complete(rest_api, headers)
    return (err is None, err)