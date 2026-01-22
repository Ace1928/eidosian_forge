import saml2
from saml2 import SamlBase
class TService_(TExtensibleDocumented_):
    """The http://schemas.xmlsoap.org/wsdl/:tService element"""
    c_tag = 'tService'
    c_namespace = NAMESPACE
    c_children = TExtensibleDocumented_.c_children.copy()
    c_attributes = TExtensibleDocumented_.c_attributes.copy()
    c_child_order = TExtensibleDocumented_.c_child_order[:]
    c_cardinality = TExtensibleDocumented_.c_cardinality.copy()
    c_children['{http://schemas.xmlsoap.org/wsdl/}port'] = ('port', [TService_port])
    c_cardinality['port'] = {'min': 0}
    c_attributes['name'] = ('name', 'NCName', True)
    c_child_order.extend(['port'])

    def __init__(self, port=None, name=None, documentation=None, text=None, extension_elements=None, extension_attributes=None):
        TExtensibleDocumented_.__init__(self, documentation=documentation, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.port = port or []
        self.name = name