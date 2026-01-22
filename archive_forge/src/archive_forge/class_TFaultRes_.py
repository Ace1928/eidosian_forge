import saml2
from saml2 import SamlBase
from saml2.schema import wsdl
class TFaultRes_(SamlBase):
    """The http://schemas.xmlsoap.org/wsdl/soap/:tFaultRes element"""
    c_tag = 'tFaultRes'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_attributes['{http://schemas.xmlsoap.org/wsdl/}required'] = ('required', 'None', False)
    c_attributes['parts'] = ('parts', 'NMTOKENS', False)
    c_attributes['encodingStyle'] = ('encoding_style', EncodingStyle_, False)
    c_attributes['use'] = ('use', UseChoice_, False)
    c_attributes['namespace'] = ('namespace', 'anyURI', False)

    def __init__(self, required=None, parts=None, encoding_style=None, use=None, namespace=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.required = required
        self.parts = parts
        self.encoding_style = encoding_style
        self.use = use
        self.namespace = namespace