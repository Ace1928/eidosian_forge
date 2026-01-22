import saml2
from saml2 import SamlBase
class SignatureProperties(SignaturePropertiesType_):
    """The http://www.w3.org/2000/09/xmldsig#:SignatureProperties element"""
    c_tag = 'SignatureProperties'
    c_namespace = NAMESPACE
    c_children = SignaturePropertiesType_.c_children.copy()
    c_attributes = SignaturePropertiesType_.c_attributes.copy()
    c_child_order = SignaturePropertiesType_.c_child_order[:]
    c_cardinality = SignaturePropertiesType_.c_cardinality.copy()