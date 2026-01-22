import saml2
from saml2 import SamlBase
class AsymmetricKeyAgreement(PublicKeyType_):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:AsymmetricKeyAgreement element"""
    c_tag = 'AsymmetricKeyAgreement'
    c_namespace = NAMESPACE
    c_children = PublicKeyType_.c_children.copy()
    c_attributes = PublicKeyType_.c_attributes.copy()
    c_child_order = PublicKeyType_.c_child_order[:]
    c_cardinality = PublicKeyType_.c_cardinality.copy()