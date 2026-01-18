from __future__ import absolute_import, division, print_function
import random
import sys
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata
from ansible.module_utils import six
from ansible.module_utils._text import to_native
def update_login_banner_message(self):
    """Update storage login banner message."""
    if self.login_banner_message:
        boundary = '---------------------------' + ''.join([str(random.randint(0, 9)) for x in range(27)])
        data_parts = list()
        data = None
        if six.PY2:
            newline = '\r\n'
            data_parts.extend(['--%s' % boundary, 'Content-Disposition: form-data; name="file"; filename="banner.txt"', 'Content-Type: text/plain', '', self.login_banner_message])
            data_parts.extend(['--%s--' % boundary, ''])
            data = newline.join(data_parts)
        else:
            newline = six.b('\r\n')
            data_parts.extend([six.b('--%s' % boundary), six.b('Content-Disposition: form-data; name="file"; filename="banner.txt"'), six.b('Content-Type: text/plain'), six.b(''), six.b(self.login_banner_message)])
            data_parts.extend([six.b('--%s--' % boundary), b''])
            data = newline.join(data_parts)
        headers = {'Content-Type': 'multipart/form-data; boundary=%s' % boundary, 'Content-Length': str(len(data))}
        try:
            rc, result = self.request('storage-systems/%s/login-banner' % self.ssid, method='POST', headers=headers, data=data)
        except Exception as err:
            self.module.fail_json(msg='Failed to set the storage system login banner message! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
    else:
        try:
            rc, result = self.request('storage-systems/%s/login-banner' % self.ssid, method='DELETE')
        except Exception as err:
            self.module.fail_json(msg='Failed to clear the storage system login banner message! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))