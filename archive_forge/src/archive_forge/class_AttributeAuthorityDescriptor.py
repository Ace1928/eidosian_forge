import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
class AttributeAuthorityDescriptor(AttributeAuthorityDescriptorType_):
    """The urn:oasis:names:tc:SAML:2.0:metadata:AttributeAuthorityDescriptor
    element"""
    c_tag = 'AttributeAuthorityDescriptor'
    c_namespace = NAMESPACE
    c_children = AttributeAuthorityDescriptorType_.c_children.copy()
    c_attributes = AttributeAuthorityDescriptorType_.c_attributes.copy()
    c_child_order = AttributeAuthorityDescriptorType_.c_child_order[:]
    c_cardinality = AttributeAuthorityDescriptorType_.c_cardinality.copy()