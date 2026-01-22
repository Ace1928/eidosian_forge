import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class RequestSecurityTokenResponse(RequestSecurityTokenResponseType_):
    """The http://docs.oasis-open.org/ws-sx/ws-trust/200512/:RequestSecurityTokenResponse element"""
    c_tag = 'RequestSecurityTokenResponse'
    c_namespace = NAMESPACE
    c_children = RequestSecurityTokenResponseType_.c_children.copy()
    c_attributes = RequestSecurityTokenResponseType_.c_attributes.copy()
    c_child_order = RequestSecurityTokenResponseType_.c_child_order[:]
    c_cardinality = RequestSecurityTokenResponseType_.c_cardinality.copy()