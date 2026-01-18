from __future__ import absolute_import, division, print_function
import time
from ansible_collections.community.crypto.plugins.module_utils.acme.utils import (
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import (
from ansible_collections.community.crypto.plugins.module_utils.acme.challenges import (
def wait_for_finalization(self, client):
    while True:
        self.refresh(client)
        if self.status in ['valid', 'invalid', 'pending', 'ready']:
            break
        time.sleep(2)
    if self.status != 'valid':
        raise ACMEProtocolException(client.module, 'Failed to wait for order to complete; got status "{status}"'.format(status=self.status), content_json=self.data)