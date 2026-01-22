import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class RequestKET(RequestKETType_):
    """The http://docs.oasis-open.org/ws-sx/ws-trust/200512/:RequestKET element"""
    c_tag = 'RequestKET'
    c_namespace = NAMESPACE
    c_children = RequestKETType_.c_children.copy()
    c_attributes = RequestKETType_.c_attributes.copy()
    c_child_order = RequestKETType_.c_child_order[:]
    c_cardinality = RequestKETType_.c_cardinality.copy()