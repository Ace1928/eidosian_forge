import saml2
from saml2 import SamlBase
class RestrictedPassword(RestrictedPasswordType_):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:RestrictedPassword element"""
    c_tag = 'RestrictedPassword'
    c_namespace = NAMESPACE
    c_children = RestrictedPasswordType_.c_children.copy()
    c_attributes = RestrictedPasswordType_.c_attributes.copy()
    c_child_order = RestrictedPasswordType_.c_child_order[:]
    c_cardinality = RestrictedPasswordType_.c_cardinality.copy()