from keystoneclient import base
import urllib.parse
def update_own_password(self, origpasswd, passwd):
    """Update password."""
    params = {'user': {'password': passwd, 'original_password': origpasswd}}
    return self._update('/OS-KSCRUD/users/%s' % self.client.user_id, params, response_key='access', method='PATCH', endpoint_filter={'interface': 'public'}, log=False)