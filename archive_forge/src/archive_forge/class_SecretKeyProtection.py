import saml2
from saml2 import SamlBase
class SecretKeyProtection(SecretKeyProtectionType_):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:SecretKeyProtection element"""
    c_tag = 'SecretKeyProtection'
    c_namespace = NAMESPACE
    c_children = SecretKeyProtectionType_.c_children.copy()
    c_attributes = SecretKeyProtectionType_.c_attributes.copy()
    c_child_order = SecretKeyProtectionType_.c_child_order[:]
    c_cardinality = SecretKeyProtectionType_.c_cardinality.copy()