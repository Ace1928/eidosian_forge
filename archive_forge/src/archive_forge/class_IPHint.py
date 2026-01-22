import saml2
from saml2 import SamlBase
from saml2 import md
class IPHint(SamlBase):
    """The urn:oasis:names:tc:SAML:metadata:ui:IPHint element"""
    c_tag = 'IPHint'
    c_namespace = NAMESPACE
    c_value_type = {'base': 'string'}
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()