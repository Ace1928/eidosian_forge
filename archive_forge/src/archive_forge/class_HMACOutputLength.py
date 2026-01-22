import saml2
from saml2 import SamlBase
class HMACOutputLength(HMACOutputLengthType_):
    c_tag = 'HMACOutputLength'
    c_namespace = NAMESPACE
    c_children = HMACOutputLengthType_.c_children.copy()
    c_attributes = HMACOutputLengthType_.c_attributes.copy()
    c_child_order = HMACOutputLengthType_.c_child_order[:]
    c_cardinality = HMACOutputLengthType_.c_cardinality.copy()