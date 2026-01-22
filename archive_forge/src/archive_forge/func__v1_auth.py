import http.client as http
import urllib.parse as urlparse
import httplib2
from keystoneclient import service_catalog as ks_service_catalog
from oslo_serialization import jsonutils
from glance.common import exception
from glance.i18n import _
def _v1_auth(self, token_url):
    creds = self.creds
    headers = {'X-Auth-User': creds['username'], 'X-Auth-Key': creds['password']}
    tenant = creds.get('tenant')
    if tenant:
        headers['X-Auth-Tenant'] = tenant
    resp, resp_body = self._do_request(token_url, 'GET', headers=headers)

    def _management_url(self, resp):
        for url_header in ('x-image-management-url', 'x-server-management-url', 'x-glance'):
            try:
                return resp[url_header]
            except KeyError as e:
                not_found = e
        raise not_found
    if resp.status in (http.OK, http.NO_CONTENT):
        try:
            if self.configure_via_auth:
                self.management_url = _management_url(self, resp)
            self.auth_token = resp['x-auth-token']
        except KeyError:
            raise exception.AuthorizationFailure()
    elif resp.status == http.USE_PROXY:
        raise exception.AuthorizationRedirect(uri=resp['location'])
    elif resp.status == http.BAD_REQUEST:
        raise exception.AuthBadRequest(url=token_url)
    elif resp.status == http.UNAUTHORIZED:
        raise exception.NotAuthenticated()
    elif resp.status == http.NOT_FOUND:
        raise exception.AuthUrlNotFound(url=token_url)
    else:
        raise Exception(_('Unexpected response: %s') % resp.status)