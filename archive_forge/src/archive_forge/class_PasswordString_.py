import saml2
from saml2 import SamlBase
class PasswordString_(AttributedString_):
    """The http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd:PasswordString element"""
    c_tag = 'PasswordString'
    c_namespace = NAMESPACE
    c_children = AttributedString_.c_children.copy()
    c_attributes = AttributedString_.c_attributes.copy()
    c_child_order = AttributedString_.c_child_order[:]
    c_cardinality = AttributedString_.c_cardinality.copy()
    c_attributes['Type'] = ('type', 'anyURI', False)

    def __init__(self, type=None, Id=None, text=None, extension_elements=None, extension_attributes=None):
        AttributedString_.__init__(self, Id=Id, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.type = type