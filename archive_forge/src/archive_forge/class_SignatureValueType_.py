import saml2
from saml2 import SamlBase
class SignatureValueType_(SamlBase):
    """The http://www.w3.org/2000/09/xmldsig#:SignatureValueType element"""
    c_tag = 'SignatureValueType'
    c_namespace = NAMESPACE
    c_value_type = {'base': 'base64Binary'}
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_attributes['Id'] = ('id', 'ID', False)

    def __init__(self, id=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.id = id