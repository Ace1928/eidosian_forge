import saml2
from saml2 import SamlBase
class TBindingOperation_fault(TBindingOperationFault_):
    c_tag = 'fault'
    c_namespace = NAMESPACE
    c_children = TBindingOperationFault_.c_children.copy()
    c_attributes = TBindingOperationFault_.c_attributes.copy()
    c_child_order = TBindingOperationFault_.c_child_order[:]
    c_cardinality = TBindingOperationFault_.c_cardinality.copy()