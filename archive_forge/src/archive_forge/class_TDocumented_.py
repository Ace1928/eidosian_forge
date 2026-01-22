import saml2
from saml2 import SamlBase
class TDocumented_(SamlBase):
    """The http://schemas.xmlsoap.org/wsdl/:tDocumented element"""
    c_tag = 'tDocumented'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://schemas.xmlsoap.org/wsdl/}documentation'] = ('documentation', TDocumented_documentation)
    c_cardinality['documentation'] = {'min': 0, 'max': 1}
    c_child_order.extend(['documentation'])

    def __init__(self, documentation=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.documentation = documentation