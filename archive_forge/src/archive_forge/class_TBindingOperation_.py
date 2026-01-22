import saml2
from saml2 import SamlBase
class TBindingOperation_(TExtensibleDocumented_):
    """The http://schemas.xmlsoap.org/wsdl/:tBindingOperation element"""
    c_tag = 'tBindingOperation'
    c_namespace = NAMESPACE
    c_children = TExtensibleDocumented_.c_children.copy()
    c_attributes = TExtensibleDocumented_.c_attributes.copy()
    c_child_order = TExtensibleDocumented_.c_child_order[:]
    c_cardinality = TExtensibleDocumented_.c_cardinality.copy()
    c_children['{http://schemas.xmlsoap.org/wsdl/}input'] = ('input', TBindingOperation_input)
    c_cardinality['input'] = {'min': 0, 'max': 1}
    c_children['{http://schemas.xmlsoap.org/wsdl/}output'] = ('output', TBindingOperation_output)
    c_cardinality['output'] = {'min': 0, 'max': 1}
    c_children['{http://schemas.xmlsoap.org/wsdl/}fault'] = ('fault', [TBindingOperation_fault])
    c_cardinality['fault'] = {'min': 0}
    c_attributes['name'] = ('name', 'NCName', True)
    c_child_order.extend(['input', 'output', 'fault'])

    def __init__(self, input=None, output=None, fault=None, name=None, documentation=None, text=None, extension_elements=None, extension_attributes=None):
        TExtensibleDocumented_.__init__(self, documentation=documentation, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.input = input
        self.output = output
        self.fault = fault or []
        self.name = name