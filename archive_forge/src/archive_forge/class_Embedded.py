import saml2
from saml2 import SamlBase
class Embedded(EmbeddedType_):
    """The http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd:Embedded element"""
    c_tag = 'Embedded'
    c_namespace = NAMESPACE
    c_children = EmbeddedType_.c_children.copy()
    c_attributes = EmbeddedType_.c_attributes.copy()
    c_child_order = EmbeddedType_.c_child_order[:]
    c_cardinality = EmbeddedType_.c_cardinality.copy()