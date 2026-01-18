from __future__ import absolute_import, division, print_function
import binascii
import os
import re
from time import sleep
from datetime import datetime
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata
from ansible.module_utils._text import to_native
def upload_certificate(self, path):
    """Add or update remote server certificate to the storage array."""
    file_name = os.path.basename(path)
    headers, data = create_multipart_formdata(files=[('file', file_name, path)])
    rc, resp = self.request(self.url_path_prefix + 'certificates/remote-server', method='POST', headers=headers, data=data, ignore_errors=True)
    if rc == 404:
        rc, resp = self.request(self.url_path_prefix + 'sslconfig/ca?useTruststore=true', method='POST', headers=headers, data=data, ignore_errors=True)
    if rc > 299:
        self.module.fail_json(msg='Failed to upload certificate. Array [%s]. Error [%s, %s].' % (self.ssid, rc, resp))