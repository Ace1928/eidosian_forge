import saml2
from saml2 import SamlBase
class TExtensibilityElement_(SamlBase):
    """The http://schemas.xmlsoap.org/wsdl/:tExtensibilityElement element"""
    c_tag = 'tExtensibilityElement'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_attributes['required'] = ('required', 'None', False)

    def __init__(self, required=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.required = required