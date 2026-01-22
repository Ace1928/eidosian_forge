import saml2
from saml2 import SamlBase
class LengthType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:LengthType element"""
    c_tag = 'LengthType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_attributes['min'] = ('min', 'integer', True)
    c_attributes['max'] = ('max', 'integer', False)

    def __init__(self, min=None, max=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.min = min
        self.max = max