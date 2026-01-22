import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
class PDPDescriptor(PDPDescriptorType_):
    """The urn:oasis:names:tc:SAML:2.0:metadata:PDPDescriptor element"""
    c_tag = 'PDPDescriptor'
    c_namespace = NAMESPACE
    c_children = PDPDescriptorType_.c_children.copy()
    c_attributes = PDPDescriptorType_.c_attributes.copy()
    c_child_order = PDPDescriptorType_.c_child_order[:]
    c_cardinality = PDPDescriptorType_.c_cardinality.copy()