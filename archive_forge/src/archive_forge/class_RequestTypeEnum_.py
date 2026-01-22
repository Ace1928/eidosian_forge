import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class RequestTypeEnum_(SamlBase):
    """The http://docs.oasis-open.org/ws-sx/ws-trust/200512/:RequestTypeEnum element"""
    c_tag = 'RequestTypeEnum'
    c_namespace = NAMESPACE
    c_value_type = {'base': 'xs:anyURI', 'enumeration': ['http://docs.oasis-open.org/ws-sx/ws-trust/200512/Issue', 'http://docs.oasis-open.org/ws-sx/ws-trust/200512/Renew', 'http://docs.oasis-open.org/ws-sx/ws-trust/200512/Cancel', 'http://docs.oasis-open.org/ws-sx/ws-trust/200512/STSCancel', 'http://docs.oasis-open.org/ws-sx/ws-trust/200512/Validate']}
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()