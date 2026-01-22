import saml2
from saml2 import SamlBase
class RelatesToType_(SamlBase):
    """The http://www.w3.org/2005/08/addressing:RelatesToType element"""
    c_tag = 'RelatesToType'
    c_namespace = NAMESPACE
    c_value_type = {'base': 'anyURI'}
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_attributes['RelationshipType'] = ('relationship_type', RelationshipTypeOpenEnum_, False)

    def __init__(self, relationship_type='http://www.w3.org/2005/08/addressing/reply', text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.relationship_type = relationship_type