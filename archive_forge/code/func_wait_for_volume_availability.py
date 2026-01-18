from __future__ import absolute_import, division, print_function
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
def wait_for_volume_availability(self, retries=VOLUME_CREATION_BLOCKING_TIMEOUT_SEC / 5):
    """Waits until volume becomes available.

        :raises AnsibleFailJson when retries are exhausted.
        """
    if retries == 0:
        self.module.fail_json(msg='Timed out waiting for the volume %s to become available. Array [%s].' % (self.name, self.ssid))
    if not self.get_volume():
        sleep(5)
        self.wait_for_volume_availability(retries=retries - 1)