import datetime
import urllib.parse
import uuid
from lxml import etree  # nosec(cjschaef): used to create xml, not parse it
from oslo_config import cfg
from keystoneclient import access
from keystoneclient.auth.identity import v3
from keystoneclient import exceptions
from keystoneclient.i18n import _
def _send_service_provider_saml2_authn_response(self, session):
    """Present SAML2 assertion to the Service Provider.

        The assertion is issued by a trusted Identity Provider for the
        authenticated user. This function directs the HTTP request to SP
        managed URL, for instance: ``https://<host>:<port>/Shibboleth.sso/
        SAML2/ECP``.
        Upon success there's a session created and access to the protected
        resource is granted. Many implementations of the SP return HTTP 302/303
        status code pointing to the protected URL (``https://<host>:<port>/v3/
        OS-FEDERATION/identity_providers/{identity_provider}/protocols/
        {protocol_id}/auth`` in this case). Saml2 plugin should point to that
        URL again, with HTTP GET method, expecting an unscoped token.

        :param session: a session object to send out HTTP requests.

        """
    self.saml2_idp_authn_response[0][0] = self.relay_state
    response = session.post(self.idp_response_consumer_url, headers=self.ECP_SP_SAML2_REQUEST_HEADERS, data=etree.tostring(self.saml2_idp_authn_response), authenticated=False, redirect=False)
    response = self._handle_http_ecp_redirect(session, response, method='GET', headers=self.ECP_SP_SAML2_REQUEST_HEADERS)
    self.authenticated_response = response