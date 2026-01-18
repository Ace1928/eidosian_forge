from __future__ import absolute_import, division, print_function
import os
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule, create_multipart_formdata
from ansible.module_utils._text import to_native, to_text, to_bytes
def upgrade_list(self):
    """Determine whether firmware is compatible with the specified drives."""
    if self.upgrade_list_cache is None:
        self.upgrade_list_cache = list()
        try:
            rc, response = self.request('storage-systems/%s/firmware/drives' % self.ssid)
            for firmware in self.firmware_list:
                filename = os.path.basename(firmware)
                for uploaded_firmware in response['compatibilities']:
                    if uploaded_firmware['filename'] == filename:
                        drive_reference_list = []
                        for drive in uploaded_firmware['compatibleDrives']:
                            try:
                                rc, drive_info = self.request('storage-systems/%s/drives/%s' % (self.ssid, drive['driveRef']))
                                if drive_info['firmwareVersion'] != uploaded_firmware['firmwareVersion'] and uploaded_firmware['firmwareVersion'] in uploaded_firmware['supportedFirmwareVersions']:
                                    if self.ignore_inaccessible_drives or (not drive_info['offline'] and drive_info['available']):
                                        drive_reference_list.append(drive['driveRef'])
                                    if not drive['onlineUpgradeCapable'] and self.upgrade_drives_online:
                                        self.module.fail_json(msg='Drive is not capable of online upgrade. Array [%s]. Drive [%s].' % (self.ssid, drive['driveRef']))
                            except Exception as error:
                                self.module.fail_json(msg='Failed to retrieve drive information. Array [%s]. Drive [%s]. Error [%s].' % (self.ssid, drive['driveRef'], to_native(error)))
                        if drive_reference_list:
                            self.upgrade_list_cache.extend([{'filename': filename, 'driveRefList': drive_reference_list}])
        except Exception as error:
            self.module.fail_json(msg='Failed to complete compatibility and health check. Array [%s]. Error [%s].' % (self.ssid, to_native(error)))
    return self.upgrade_list_cache