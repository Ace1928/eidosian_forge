import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class DelegateTo(DelegateToType_):
    """The http://docs.oasis-open.org/ws-sx/ws-trust/200512/:DelegateTo element"""
    c_tag = 'DelegateTo'
    c_namespace = NAMESPACE
    c_children = DelegateToType_.c_children.copy()
    c_attributes = DelegateToType_.c_attributes.copy()
    c_child_order = DelegateToType_.c_child_order[:]
    c_cardinality = DelegateToType_.c_cardinality.copy()