import saml2
from saml2 import SamlBase
class TMessage_part(TPart_):
    c_tag = 'part'
    c_namespace = NAMESPACE
    c_children = TPart_.c_children.copy()
    c_attributes = TPart_.c_attributes.copy()
    c_child_order = TPart_.c_child_order[:]
    c_cardinality = TPart_.c_cardinality.copy()