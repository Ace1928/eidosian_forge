import saml2
from saml2 import SamlBase
class TPortType_(TExtensibleAttributesDocumented_):
    """The http://schemas.xmlsoap.org/wsdl/:tPortType element"""
    c_tag = 'tPortType'
    c_namespace = NAMESPACE
    c_children = TExtensibleAttributesDocumented_.c_children.copy()
    c_attributes = TExtensibleAttributesDocumented_.c_attributes.copy()
    c_child_order = TExtensibleAttributesDocumented_.c_child_order[:]
    c_cardinality = TExtensibleAttributesDocumented_.c_cardinality.copy()
    c_children['{http://schemas.xmlsoap.org/wsdl/}operation'] = ('operation', [TPortType_operation])
    c_cardinality['operation'] = {'min': 0}
    c_attributes['name'] = ('name', 'NCName', True)
    c_child_order.extend(['operation'])

    def __init__(self, operation=None, name=None, documentation=None, text=None, extension_elements=None, extension_attributes=None):
        TExtensibleAttributesDocumented_.__init__(self, documentation=documentation, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.operation = operation or []
        self.name = name