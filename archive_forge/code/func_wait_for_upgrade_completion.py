from __future__ import absolute_import, division, print_function
import os
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule, create_multipart_formdata
from ansible.module_utils._text import to_native, to_text, to_bytes
def wait_for_upgrade_completion(self):
    """Wait for drive firmware upgrade to complete."""
    drive_references = [reference for drive in self.upgrade_list() for reference in drive['driveRefList']]
    last_status = None
    for attempt in range(int(self.WAIT_TIMEOUT_SEC / 5)):
        try:
            rc, response = self.request('storage-systems/%s/firmware/drives/state' % self.ssid)
            for status in response['driveStatus']:
                last_status = status
                if status['driveRef'] in drive_references:
                    if status['status'] == 'okay':
                        continue
                    elif status['status'] in ['inProgress', 'inProgressRecon', 'pending', 'notAttempted']:
                        break
                    else:
                        self.module.fail_json(msg='Drive firmware upgrade failed. Array [%s]. Drive [%s]. Status [%s].' % (self.ssid, status['driveRef'], status['status']))
            else:
                self.upgrade_in_progress = False
                break
        except Exception as error:
            self.module.fail_json(msg='Failed to retrieve drive status. Array [%s]. Error [%s].' % (self.ssid, to_native(error)))
        sleep(5)
    else:
        self.module.fail_json(msg='Timed out waiting for drive firmware upgrade. Array [%s]. Status [%s].' % (self.ssid, last_status))