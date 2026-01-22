import saml2
from saml2 import SamlBase
from saml2 import xmldsig as ds
class EncryptionProperty(EncryptionPropertyType_):
    """The http://www.w3.org/2001/04/xmlenc#:EncryptionProperty element"""
    c_tag = 'EncryptionProperty'
    c_namespace = NAMESPACE
    c_children = EncryptionPropertyType_.c_children.copy()
    c_attributes = EncryptionPropertyType_.c_attributes.copy()
    c_child_order = EncryptionPropertyType_.c_child_order[:]
    c_cardinality = EncryptionPropertyType_.c_cardinality.copy()