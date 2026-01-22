from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.identity.v3 import federation
from keystoneauth1 import plugin
def _get_ecp_assertion(self, session):
    body = {'auth': {'identity': {'methods': ['token'], 'token': {'id': self._local_cloud_plugin.get_token(session)}}, 'scope': {'service_provider': {'id': self._sp_id}}}}
    endpoint_filter = {'version': (3, 0), 'interface': plugin.AUTH_INTERFACE}
    headers = {'Accept': 'application/json'}
    resp = session.post(self.REQUEST_ECP_URL, json=body, auth=self._local_cloud_plugin, endpoint_filter=endpoint_filter, headers=headers, authenticated=False, raise_exc=False)
    if not resp.ok:
        msg = 'Error while requesting ECP wrapped assertion: response exit code: %(status_code)d, reason: %(err)s'
        msg = msg % {'status_code': resp.status_code, 'err': resp.reason}
        raise exceptions.AuthorizationFailure(msg)
    if not resp.text:
        raise exceptions.InvalidResponse(resp)
    return str(resp.text)