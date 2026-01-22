import saml2
from saml2 import SamlBase
from saml2 import saml
class EntityAttributes(EntityAttributesType_):
    """The urn:oasis:names:tc:SAML:metadata:attribute:EntityAttributes element"""
    c_tag = 'EntityAttributes'
    c_namespace = NAMESPACE
    c_children = EntityAttributesType_.c_children.copy()
    c_attributes = EntityAttributesType_.c_attributes.copy()
    c_child_order = EntityAttributesType_.c_child_order[:]
    c_cardinality = EntityAttributesType_.c_cardinality.copy()