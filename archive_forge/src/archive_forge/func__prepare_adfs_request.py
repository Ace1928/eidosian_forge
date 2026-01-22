import datetime
import urllib
import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.extras._saml2.v3 import base
def _prepare_adfs_request(self):
    """Build the ADFS Request Security Token SOAP message.

        Some values like username or password are inserted in the request.

        """
    WSS_SECURITY_NAMESPACE = {'o': 'http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd'}
    TRUST_NAMESPACE = {'trust': 'http://docs.oasis-open.org/ws-sx/ws-trust/200512'}
    WSP_NAMESPACE = {'wsp': 'http://schemas.xmlsoap.org/ws/2004/09/policy'}
    WSA_NAMESPACE = {'wsa': 'http://www.w3.org/2005/08/addressing'}
    root = etree.Element('{http://www.w3.org/2003/05/soap-envelope}Envelope', nsmap=self.NAMESPACES)
    header = etree.SubElement(root, '{http://www.w3.org/2003/05/soap-envelope}Header')
    action = etree.SubElement(header, '{http://www.w3.org/2005/08/addressing}Action')
    action.set('{http://www.w3.org/2003/05/soap-envelope}mustUnderstand', '1')
    action.text = 'http://docs.oasis-open.org/ws-sx/ws-trust/200512/RST/Issue'
    messageID = etree.SubElement(header, '{http://www.w3.org/2005/08/addressing}MessageID')
    messageID.text = 'urn:uuid:' + uuid.uuid4().hex
    replyID = etree.SubElement(header, '{http://www.w3.org/2005/08/addressing}ReplyTo')
    address = etree.SubElement(replyID, '{http://www.w3.org/2005/08/addressing}Address')
    address.text = 'http://www.w3.org/2005/08/addressing/anonymous'
    to = etree.SubElement(header, '{http://www.w3.org/2005/08/addressing}To')
    to.set('{http://www.w3.org/2003/05/soap-envelope}mustUnderstand', '1')
    security = etree.SubElement(header, '{http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd}Security', nsmap=WSS_SECURITY_NAMESPACE)
    security.set('{http://www.w3.org/2003/05/soap-envelope}mustUnderstand', '1')
    timestamp = etree.SubElement(security, '{http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd}Timestamp')
    timestamp.set('{http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd}Id', '_0')
    created = etree.SubElement(timestamp, '{http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd}Created')
    expires = etree.SubElement(timestamp, '{http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd}Expires')
    created.text, expires.text = self._token_dates()
    usernametoken = etree.SubElement(security, '{http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd}UsernameToken')
    usernametoken.set('{http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd}u', 'uuid-%s-1' % uuid.uuid4().hex)
    username = etree.SubElement(usernametoken, '{http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd}Username')
    password = etree.SubElement(usernametoken, '{http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd}Password', Type='http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordText')
    body = etree.SubElement(root, '{http://www.w3.org/2003/05/soap-envelope}Body')
    request_security_token = etree.SubElement(body, '{http://docs.oasis-open.org/ws-sx/ws-trust/200512}RequestSecurityToken', nsmap=TRUST_NAMESPACE)
    applies_to = etree.SubElement(request_security_token, '{http://schemas.xmlsoap.org/ws/2004/09/policy}AppliesTo', nsmap=WSP_NAMESPACE)
    endpoint_reference = etree.SubElement(applies_to, '{http://www.w3.org/2005/08/addressing}EndpointReference', nsmap=WSA_NAMESPACE)
    wsa_address = etree.SubElement(endpoint_reference, '{http://www.w3.org/2005/08/addressing}Address')
    keytype = etree.SubElement(request_security_token, '{http://docs.oasis-open.org/ws-sx/ws-trust/200512}KeyType')
    keytype.text = 'http://docs.oasis-open.org/ws-sx/ws-trust/200512/Bearer'
    request_type = etree.SubElement(request_security_token, '{http://docs.oasis-open.org/ws-sx/ws-trust/200512}RequestType')
    request_type.text = 'http://docs.oasis-open.org/ws-sx/ws-trust/200512/Issue'
    token_type = etree.SubElement(request_security_token, '{http://docs.oasis-open.org/ws-sx/ws-trust/200512}TokenType')
    token_type.text = 'urn:oasis:names:tc:SAML:1.0:assertion'
    username.text = self.username
    password.text = self.password
    to.text = self.identity_provider_url
    wsa_address.text = self.service_provider_entity_id or self.service_provider_endpoint
    self.prepared_request = root