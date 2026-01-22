import saml2
from saml2 import SamlBase
class Fault_faultcode(SamlBase):
    c_tag = 'faultcode'
    c_namespace = NAMESPACE
    c_value_type = {'base': 'QName'}
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()