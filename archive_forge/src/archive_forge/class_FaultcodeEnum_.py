import saml2
from saml2 import SamlBase
class FaultcodeEnum_(SamlBase):
    """The http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd:FaultcodeEnum element"""
    c_tag = 'FaultcodeEnum'
    c_namespace = NAMESPACE
    c_value_type = {'base': 'xsd:QName', 'enumeration': ['wsse:UnsupportedSecurityToken', 'wsse:UnsupportedAlgorithm', 'wsse:InvalidSecurity', 'wsse:InvalidSecurityToken', 'wsse:FailedAuthentication', 'wsse:FailedCheck', 'wsse:SecurityTokenUnavailable']}
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()