import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
class EntitiesDescriptor(EntitiesDescriptorType_):
    """The urn:oasis:names:tc:SAML:2.0:metadata:EntitiesDescriptor element"""
    c_tag = 'EntitiesDescriptor'
    c_namespace = NAMESPACE
    c_children = EntitiesDescriptorType_.c_children.copy()
    c_attributes = EntitiesDescriptorType_.c_attributes.copy()
    c_child_order = EntitiesDescriptorType_.c_child_order[:]
    c_cardinality = EntitiesDescriptorType_.c_cardinality.copy()