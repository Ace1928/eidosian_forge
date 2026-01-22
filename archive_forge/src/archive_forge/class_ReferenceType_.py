import saml2
from saml2 import SamlBase
class ReferenceType_(SamlBase):
    """The http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd:ReferenceType element"""
    c_tag = 'ReferenceType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_attributes['URI'] = ('uri', 'anyURI', False)
    c_attributes['ValueType'] = ('value_type', 'anyURI', False)

    def __init__(self, uri=None, value_type=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.uri = uri
        self.value_type = value_type