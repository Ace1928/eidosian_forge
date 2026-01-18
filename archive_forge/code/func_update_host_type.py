from __future__ import absolute_import, division, print_function
import random
import sys
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata
from ansible.module_utils import six
from ansible.module_utils._text import to_native
def update_host_type(self):
    """Update default host type."""
    try:
        rc, default_host_type = self.request('storage-systems/%s/symbol/setStorageArrayProperties?verboseErrorResponse=true' % self.ssid, method='POST', data={'settings': {'defaultHostTypeIndex': self.host_type_index}})
    except Exception as error:
        self.module.fail_json(msg='Failed to set default host type. Array [%s]. Error [%s]' % (self.ssid, to_native(error)))