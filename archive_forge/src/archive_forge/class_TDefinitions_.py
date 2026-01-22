import saml2
from saml2 import SamlBase
class TDefinitions_(TExtensibleDocumented_):
    """The http://schemas.xmlsoap.org/wsdl/:tDefinitions element"""
    c_tag = 'tDefinitions'
    c_namespace = NAMESPACE
    c_children = TExtensibleDocumented_.c_children.copy()
    c_attributes = TExtensibleDocumented_.c_attributes.copy()
    c_child_order = TExtensibleDocumented_.c_child_order[:]
    c_cardinality = TExtensibleDocumented_.c_cardinality.copy()
    c_children['{http://schemas.xmlsoap.org/wsdl/}import'] = ('import', Import)
    c_cardinality['import'] = {'min': 0, 'max': 1}
    c_children['{http://schemas.xmlsoap.org/wsdl/}types'] = ('types', Types)
    c_cardinality['types'] = {'min': 0, 'max': 1}
    c_children['{http://schemas.xmlsoap.org/wsdl/}message'] = ('message', Message)
    c_cardinality['message'] = {'min': 0, 'max': 1}
    c_children['{http://schemas.xmlsoap.org/wsdl/}portType'] = ('port_type', PortType)
    c_cardinality['port_type'] = {'min': 0, 'max': 1}
    c_children['{http://schemas.xmlsoap.org/wsdl/}binding'] = ('binding', Binding)
    c_cardinality['binding'] = {'min': 0, 'max': 1}
    c_children['{http://schemas.xmlsoap.org/wsdl/}service'] = ('service', Service)
    c_cardinality['service'] = {'min': 0, 'max': 1}
    c_attributes['targetNamespace'] = ('target_namespace', 'anyURI', False)
    c_attributes['name'] = ('name', 'NCName', False)
    c_child_order.extend(['import', 'types', 'message', 'port_type', 'binding', 'service'])

    def __init__(self, import_=None, types=None, message=None, port_type=None, binding=None, service=None, target_namespace=None, name=None, documentation=None, text=None, extension_elements=None, extension_attributes=None):
        TExtensibleDocumented_.__init__(self, documentation=documentation, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.import_ = import_
        self.types = types
        self.message = message
        self.port_type = port_type
        self.binding = binding
        self.service = service
        self.target_namespace = target_namespace
        self.name = name