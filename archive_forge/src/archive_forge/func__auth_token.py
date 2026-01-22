from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
def _auth_token(self):
    auth = GcpSession(self.module, 'auth')
    response = auth.get(self_link(self.module))
    self.fetch = response.json()
    return response.request.headers['authorization'].split(' ')[1]