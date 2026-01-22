import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class StatusType_Code(StatusCodeOpenEnum_):
    c_tag = 'Code'
    c_namespace = NAMESPACE
    c_children = StatusCodeOpenEnum_.c_children.copy()
    c_attributes = StatusCodeOpenEnum_.c_attributes.copy()
    c_child_order = StatusCodeOpenEnum_.c_child_order[:]
    c_cardinality = StatusCodeOpenEnum_.c_cardinality.copy()