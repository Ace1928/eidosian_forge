import saml2
from saml2 import SamlBase
class SecurityTokenReference(SecurityTokenReferenceType_):
    """The http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd:SecurityTokenReference element"""
    c_tag = 'SecurityTokenReference'
    c_namespace = NAMESPACE
    c_children = SecurityTokenReferenceType_.c_children.copy()
    c_attributes = SecurityTokenReferenceType_.c_attributes.copy()
    c_child_order = SecurityTokenReferenceType_.c_child_order[:]
    c_cardinality = SecurityTokenReferenceType_.c_cardinality.copy()