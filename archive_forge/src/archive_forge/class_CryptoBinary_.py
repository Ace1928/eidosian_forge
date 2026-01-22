import saml2
from saml2 import SamlBase
class CryptoBinary_(SamlBase):
    """The http://www.w3.org/2000/09/xmldsig#:CryptoBinary element"""
    c_tag = 'CryptoBinary'
    c_namespace = NAMESPACE
    c_value_type = {'base': 'base64Binary'}
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()