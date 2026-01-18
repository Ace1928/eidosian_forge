from __future__ import absolute_import, division, print_function
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
from time import sleep
def update_minimum_password_length(self):
    """Update automatic load balancing state."""
    try:
        if self.is_proxy():
            if self.ssid == '0' or self.ssid.lower() == 'proxy':
                try:
                    if not self.is_admin_password_set:
                        self.creds['url_password'] = 'admin'
                    rc, minimum_password_length = self.request('local-users/password-length', method='POST', data={'minimumPasswordLength': self.minimum_password_length})
                except Exception as error:
                    if not self.is_admin_password_set:
                        self.creds['url_password'] = ''
                    rc, minimum_password_length = self.request('local-users/password-length', method='POST', data={'minimumPasswordLength': self.minimum_password_length})
            elif self.is_embedded_available():
                if not self.is_admin_password_set:
                    self.creds['url_password'] = ''
                rc, minimum_password_length = self.request('storage-systems/%s/forward/devmgr/v2/storage-systems/1/local-users/password-length' % self.ssid, method='POST', data={'minimumPasswordLength': self.minimum_password_length})
        else:
            if not self.is_admin_password_set:
                self.creds['url_password'] = ''
            rc, minimum_password_length = self.request('storage-systems/%s/local-users/password-length' % self.ssid, method='POST', data={'minimumPasswordLength': self.minimum_password_length})
    except Exception as error:
        self.module.fail_json(msg='Failed to set minimum password length. Array [%s]. Error [%s].' % (self.ssid, to_native(error)))