import datetime
import urllib
import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.extras._saml2.v3 import base
def _access_service_provider(self, session):
    """Access protected endpoint and fetch unscoped token.

        After federated authentication workflow a protected endpoint should be
        accessible with the session object. The access is granted basing on the
        cookies stored within the session object. If, for some reason no
        cookies are present (quantity test) it means something went wrong and
        user will not be able to fetch an unscoped token. In that case an
        ``exceptions.AuthorizationFailure` exception is raised and no HTTP call
        is even made.

        :param session : a session object to send out HTTP requests.
        :type session: keystoneauth1.session.Session

        :raises keystoneauth1.exceptions.AuthorizationFailure: in case session
        object has empty cookie jar.

        """
    if self._cookies(session) is False:
        raise exceptions.AuthorizationFailure("Session object doesn't contain a cookie, therefore you are not allowed to enter the Identity Provider's protected area.")
    self.authenticated_response = session.get(self.federated_token_url, authenticated=False)