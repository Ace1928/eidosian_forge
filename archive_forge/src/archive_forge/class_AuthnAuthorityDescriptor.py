import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
class AuthnAuthorityDescriptor(AuthnAuthorityDescriptorType_):
    """The urn:oasis:names:tc:SAML:2.0:metadata:AuthnAuthorityDescriptor
    element"""
    c_tag = 'AuthnAuthorityDescriptor'
    c_namespace = NAMESPACE
    c_children = AuthnAuthorityDescriptorType_.c_children.copy()
    c_attributes = AuthnAuthorityDescriptorType_.c_attributes.copy()
    c_child_order = AuthnAuthorityDescriptorType_.c_child_order[:]
    c_cardinality = AuthnAuthorityDescriptorType_.c_cardinality.copy()