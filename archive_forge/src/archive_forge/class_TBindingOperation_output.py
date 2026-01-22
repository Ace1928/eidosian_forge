import saml2
from saml2 import SamlBase
class TBindingOperation_output(TBindingOperationMessage_):
    c_tag = 'output'
    c_namespace = NAMESPACE
    c_children = TBindingOperationMessage_.c_children.copy()
    c_attributes = TBindingOperationMessage_.c_attributes.copy()
    c_child_order = TBindingOperationMessage_.c_child_order[:]
    c_cardinality = TBindingOperationMessage_.c_cardinality.copy()