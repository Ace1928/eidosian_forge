import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class BinaryExchange(BinaryExchangeType_):
    """The http://docs.oasis-open.org/ws-sx/ws-trust/200512/:BinaryExchange element"""
    c_tag = 'BinaryExchange'
    c_namespace = NAMESPACE
    c_children = BinaryExchangeType_.c_children.copy()
    c_attributes = BinaryExchangeType_.c_attributes.copy()
    c_child_order = BinaryExchangeType_.c_child_order[:]
    c_cardinality = BinaryExchangeType_.c_cardinality.copy()