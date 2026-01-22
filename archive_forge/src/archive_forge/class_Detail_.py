import saml2
from saml2 import SamlBase
class Detail_(SamlBase):
    """The http://schemas.xmlsoap.org/soap/envelope/:detail element"""
    c_tag = 'detail'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()