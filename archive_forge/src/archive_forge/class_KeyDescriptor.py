import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
class KeyDescriptor(KeyDescriptorType_):
    """The urn:oasis:names:tc:SAML:2.0:metadata:KeyDescriptor element"""
    c_tag = 'KeyDescriptor'
    c_namespace = NAMESPACE
    c_children = KeyDescriptorType_.c_children.copy()
    c_attributes = KeyDescriptorType_.c_attributes.copy()
    c_child_order = KeyDescriptorType_.c_child_order[:]
    c_cardinality = KeyDescriptorType_.c_cardinality.copy()