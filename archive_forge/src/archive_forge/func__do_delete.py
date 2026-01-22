from manilaclient import api_versions
from manilaclient import base
def _do_delete(self, tenant_id, user_id=None, share_type=None, resource_path=RESOURCE_PATH):
    self._check_user_id_and_share_type_args(user_id, share_type)
    data = {'resource_path': resource_path, 'tenant_id': tenant_id, 'user_id': user_id, 'st': share_type}
    if user_id:
        url = '%(resource_path)s/%(tenant_id)s?user_id=%(user_id)s' % data
    elif share_type:
        url = '%(resource_path)s/%(tenant_id)s?share_type=%(st)s' % data
    else:
        url = '%(resource_path)s/%(tenant_id)s' % data
    self._delete(url)