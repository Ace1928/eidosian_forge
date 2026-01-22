import saml2
from saml2 import SamlBase
class KeySharing(KeySharingType_):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:KeySharing element"""
    c_tag = 'KeySharing'
    c_namespace = NAMESPACE
    c_children = KeySharingType_.c_children.copy()
    c_attributes = KeySharingType_.c_attributes.copy()
    c_child_order = KeySharingType_.c_child_order[:]
    c_cardinality = KeySharingType_.c_cardinality.copy()