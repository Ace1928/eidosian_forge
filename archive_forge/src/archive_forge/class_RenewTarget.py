import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class RenewTarget(RenewTargetType_):
    """The http://docs.oasis-open.org/ws-sx/ws-trust/200512/:RenewTarget element"""
    c_tag = 'RenewTarget'
    c_namespace = NAMESPACE
    c_children = RenewTargetType_.c_children.copy()
    c_attributes = RenewTargetType_.c_attributes.copy()
    c_child_order = RenewTargetType_.c_child_order[:]
    c_cardinality = RenewTargetType_.c_cardinality.copy()