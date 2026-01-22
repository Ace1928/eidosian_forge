import saml2
from saml2 import SamlBase
class TPortType_operation(TOperation_):
    c_tag = 'operation'
    c_namespace = NAMESPACE
    c_children = TOperation_.c_children.copy()
    c_attributes = TOperation_.c_attributes.copy()
    c_child_order = TOperation_.c_child_order[:]
    c_cardinality = TOperation_.c_cardinality.copy()