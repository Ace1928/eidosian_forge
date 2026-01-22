import datetime
import urllib
import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.extras._saml2.v3 import base
def _get_adfs_security_token(self, session):
    """Send ADFS Security token to the ADFS server.

        Store the result in the instance attribute and raise an exception in
        case the response is not valid XML data.

        If a user cannot authenticate due to providing bad credentials, the
        ADFS2.0 server will return a HTTP 500 response and a XML Fault message.
        If ``exceptions.InternalServerError`` is caught, the method tries to
        parse the XML response.
        If parsing is unsuccessful, an ``exceptions.AuthorizationFailure`` is
        raised with a reason from the XML fault. Otherwise an original
        ``exceptions.InternalServerError`` is re-raised.

        :param session : a session object to send out HTTP requests.
        :type session: keystoneauth1.session.Session

        :raises keystoneauth1.exceptions.AuthorizationFailure: when HTTP
                 response from the ADFS server is not a valid XML ADFS security
                 token.
        :raises keystoneauth1.exceptions.InternalServerError: If response
                 status code is HTTP 500 and the response XML cannot be
                 recognized.

        """

    def _get_failure(e):
        xpath = '/s:Envelope/s:Body/s:Fault/s:Code/s:Subcode/s:Value'
        content = e.response.content
        try:
            obj = self.str_to_xml(content).xpath(xpath, namespaces=self.NAMESPACES)
            obj = self._first(obj)
            return obj.text
        except (IndexError, exceptions.AuthorizationFailure):
            raise e
    request_security_token = self.xml_to_str(self.prepared_request)
    try:
        response = session.post(url=self.identity_provider_url, headers=self.HEADER_SOAP, data=request_security_token, authenticated=False)
    except exceptions.InternalServerError as e:
        reason = _get_failure(e)
        raise exceptions.AuthorizationFailure(reason)
    msg = 'Error parsing XML returned from the ADFS Identity Provider, reason: %s'
    self.adfs_token = self.str_to_xml(response.content, msg)