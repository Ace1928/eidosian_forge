import saml2
from saml2 import SamlBase
class OperatorContentType_(SamlBase):
    """The http://schemas.xmlsoap.org/ws/2004/09/policy:OperatorContentType element"""
    c_tag = 'OperatorContentType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_cardinality['policy'] = {'min': 0}
    c_cardinality['all'] = {'min': 0}
    c_cardinality['exactly_one'] = {'min': 0}
    c_children['{http://schemas.xmlsoap.org/ws/2004/09/policy}PolicyReference'] = ('policy_reference', [PolicyReference])
    c_cardinality['policy_reference'] = {'min': 0}
    c_child_order.extend(['policy', 'all', 'exactly_one', 'policy_reference'])

    def __init__(self, policy=None, all=None, exactly_one=None, policy_reference=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.policy = policy or []
        self.all = all or []
        self.exactly_one = exactly_one or []
        self.policy_reference = policy_reference or []