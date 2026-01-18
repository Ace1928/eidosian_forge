from __future__ import absolute_import, division, print_function
import os
import multiprocessing
import threading
from time import sleep
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata, request
from ansible.module_utils._text import to_native
def wait_for_reboot(self):
    """Wait for controller A to fully reboot and web services running"""
    reboot_started = False
    reboot_completed = False
    self.module.log('Controller firmware: Reboot commencing. Array Id [%s].' % self.ssid)
    while self.wait_for_completion and (not (reboot_started and reboot_completed)):
        try:
            rc, response = self.request('storage-systems/%s/symbol/pingController?controller=a&verboseErrorResponse=true' % self.ssid, method='POST', timeout=10, log_request=False)
            if reboot_started and response == 'ok':
                self.module.log('Controller firmware: Reboot completed. Array Id [%s].' % self.ssid)
                reboot_completed = True
            sleep(2)
        except Exception as error:
            if not reboot_started:
                self.module.log('Controller firmware: Reboot started. Array Id [%s].' % self.ssid)
                reboot_started = True
            continue