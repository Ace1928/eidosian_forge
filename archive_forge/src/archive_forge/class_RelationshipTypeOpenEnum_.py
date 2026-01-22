import saml2
from saml2 import SamlBase
class RelationshipTypeOpenEnum_(SamlBase):
    """The http://www.w3.org/2005/08/addressing:RelationshipTypeOpenEnum element"""
    c_tag = 'RelationshipTypeOpenEnum'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()