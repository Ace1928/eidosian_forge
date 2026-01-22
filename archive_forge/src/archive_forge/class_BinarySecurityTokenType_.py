import saml2
from saml2 import SamlBase
class BinarySecurityTokenType_(EncodedString_):
    """The http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd:BinarySecurityTokenType element"""
    c_tag = 'BinarySecurityTokenType'
    c_namespace = NAMESPACE
    c_children = EncodedString_.c_children.copy()
    c_attributes = EncodedString_.c_attributes.copy()
    c_child_order = EncodedString_.c_child_order[:]
    c_cardinality = EncodedString_.c_cardinality.copy()
    c_attributes['ValueType'] = ('value_type', 'anyURI', False)

    def __init__(self, value_type=None, encoding_type=None, Id=None, text=None, extension_elements=None, extension_attributes=None):
        EncodedString_.__init__(self, encoding_type=encoding_type, Id=Id, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.value_type = value_type