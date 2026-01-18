from __future__ import absolute_import, division, print_function
import json
import threading
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
from time import sleep
def update_system(self, system):
    """Update storage system configuration."""
    try:
        rc, storage_system = self.request('storage-systems/%s' % system['ssid'], method='POST', data=system['changes'])
    except Exception as error:
        self.module.warn('Failed to update storage system. Array [%s]. Error [%s]' % (system['ssid'], to_native(error)))