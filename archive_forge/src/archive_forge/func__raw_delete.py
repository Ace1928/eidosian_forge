import copy
from openstack import exceptions
from openstack.object_store.v1 import _base
from openstack import resource
def _raw_delete(self, session, microversion=None):
    if not self.allow_delete:
        raise exceptions.MethodNotSupported(self, 'delete')
    request = self._prepare_request()
    session = self._get_session(session)
    if microversion is None:
        microversion = self._get_microversion(session, action='delete')
    if self.is_static_large_object is None:
        self.head(session)
    headers = {}
    if self.is_static_large_object:
        headers['multipart-manifest'] = 'delete'
    return session.delete(request.url, headers=headers, microversion=microversion)