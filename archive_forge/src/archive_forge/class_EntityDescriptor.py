import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
class EntityDescriptor(EntityDescriptorType_):
    """The urn:oasis:names:tc:SAML:2.0:metadata:EntityDescriptor element"""
    c_tag = 'EntityDescriptor'
    c_namespace = NAMESPACE
    c_children = EntityDescriptorType_.c_children.copy()
    c_attributes = EntityDescriptorType_.c_attributes.copy()
    c_child_order = EntityDescriptorType_.c_child_order[:]
    c_cardinality = EntityDescriptorType_.c_cardinality.copy()