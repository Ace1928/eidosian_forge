import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class RequestedSecurityToken(RequestedSecurityTokenType_):
    """The http://docs.oasis-open.org/ws-sx/ws-trust/200512/:RequestedSecurityToken element"""
    c_tag = 'RequestedSecurityToken'
    c_namespace = NAMESPACE
    c_children = RequestedSecurityTokenType_.c_children.copy()
    c_attributes = RequestedSecurityTokenType_.c_attributes.copy()
    c_child_order = RequestedSecurityTokenType_.c_child_order[:]
    c_cardinality = RequestedSecurityTokenType_.c_cardinality.copy()