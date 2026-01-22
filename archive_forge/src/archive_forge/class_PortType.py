import saml2
from saml2 import SamlBase
class PortType(TPortType_):
    """The http://schemas.xmlsoap.org/wsdl/:portType element"""
    c_tag = 'portType'
    c_namespace = NAMESPACE
    c_children = TPortType_.c_children.copy()
    c_attributes = TPortType_.c_attributes.copy()
    c_child_order = TPortType_.c_child_order[:]
    c_cardinality = TPortType_.c_cardinality.copy()