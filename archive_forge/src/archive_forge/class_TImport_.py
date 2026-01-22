import saml2
from saml2 import SamlBase
class TImport_(TExtensibleAttributesDocumented_):
    """The http://schemas.xmlsoap.org/wsdl/:tImport element"""
    c_tag = 'tImport'
    c_namespace = NAMESPACE
    c_children = TExtensibleAttributesDocumented_.c_children.copy()
    c_attributes = TExtensibleAttributesDocumented_.c_attributes.copy()
    c_child_order = TExtensibleAttributesDocumented_.c_child_order[:]
    c_cardinality = TExtensibleAttributesDocumented_.c_cardinality.copy()
    c_attributes['namespace'] = ('namespace', 'anyURI', True)
    c_attributes['location'] = ('location', 'anyURI', True)

    def __init__(self, namespace=None, location=None, documentation=None, text=None, extension_elements=None, extension_attributes=None):
        TExtensibleAttributesDocumented_.__init__(self, documentation=documentation, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.namespace = namespace
        self.location = location