from __future__ import absolute_import, division, print_function
import datetime
import re
import time
import tarfile
from ansible.module_utils.urls import fetch_file
from ansible_collections.community.general.plugins.module_utils.redfish_utils import RedfishUtils
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlunparse
def _get_installed_firmware_version_of_multi_tenant_system(self, iom_a_firmware_version, iom_b_firmware_version):
    """Return the version for the active IOM on a multi-tenant system.

        Only call this on a multi-tenant system.
        Given the installed firmware versions for IOM A, B, this method will determine which IOM is active
        for this tenanat, and return that IOM's firmware version.
        """
    which_iom_is_this = None
    for iom_letter in ['A', 'B']:
        iom_uri = 'Chassis/IOModule{0}FRU'.format(iom_letter)
        response = self.get_request(self.root_uri + self.service_root + iom_uri)
        if response['ret'] is False:
            continue
        data = response['data']
        if 'Id' in data:
            which_iom_is_this = iom_letter
            break
    if which_iom_is_this == 'A':
        return iom_a_firmware_version
    elif which_iom_is_this == 'B':
        return iom_b_firmware_version
    else:
        return None