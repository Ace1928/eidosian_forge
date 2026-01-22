import saml2
from saml2 import SamlBase
class ProblemAction(ProblemActionType_):
    """The http://www.w3.org/2005/08/addressing:ProblemAction element"""
    c_tag = 'ProblemAction'
    c_namespace = NAMESPACE
    c_children = ProblemActionType_.c_children.copy()
    c_attributes = ProblemActionType_.c_attributes.copy()
    c_child_order = ProblemActionType_.c_child_order[:]
    c_cardinality = ProblemActionType_.c_cardinality.copy()