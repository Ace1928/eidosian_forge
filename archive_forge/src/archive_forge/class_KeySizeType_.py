import saml2
from saml2 import SamlBase
from saml2 import xmldsig as ds
class KeySizeType_(SamlBase):
    """The http://www.w3.org/2001/04/xmlenc#:KeySizeType element"""
    c_tag = 'KeySizeType'
    c_namespace = NAMESPACE
    c_value_type = {'base': 'integer'}
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()