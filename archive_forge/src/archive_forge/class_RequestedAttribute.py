import saml2
from saml2 import SamlBase
from saml2 import saml
class RequestedAttribute(RequestedAttributeType_):
    """The http://eidas.europa.eu/saml-extensions:RequestedAttribute element"""
    c_tag = 'RequestedAttribute'
    c_namespace = NAMESPACE
    c_children = RequestedAttributeType_.c_children.copy()
    c_attributes = RequestedAttributeType_.c_attributes.copy()
    c_child_order = RequestedAttributeType_.c_child_order[:]
    c_cardinality = RequestedAttributeType_.c_cardinality.copy()