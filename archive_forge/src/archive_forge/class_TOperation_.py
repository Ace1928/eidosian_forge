import saml2
from saml2 import SamlBase
from saml2.schema import wsdl
class TOperation_(wsdl.TExtensibilityElement_):
    """The http://schemas.xmlsoap.org/wsdl/soap/:tOperation element"""
    c_tag = 'tOperation'
    c_namespace = NAMESPACE
    c_children = wsdl.TExtensibilityElement_.c_children.copy()
    c_attributes = wsdl.TExtensibilityElement_.c_attributes.copy()
    c_child_order = wsdl.TExtensibilityElement_.c_child_order[:]
    c_cardinality = wsdl.TExtensibilityElement_.c_cardinality.copy()
    c_attributes['soapAction'] = ('soap_action', 'anyURI', False)
    c_attributes['style'] = ('style', TStyleChoice_, False)

    def __init__(self, soap_action=None, style=None, required=None, text=None, extension_elements=None, extension_attributes=None):
        wsdl.TExtensibilityElement_.__init__(self, required=required, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.soap_action = soap_action
        self.style = style