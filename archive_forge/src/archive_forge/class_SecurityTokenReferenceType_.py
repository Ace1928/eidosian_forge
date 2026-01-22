import saml2
from saml2 import SamlBase
class SecurityTokenReferenceType_(SamlBase):
    """The http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd:SecurityTokenReferenceType element"""
    c_tag = 'SecurityTokenReferenceType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_attributes['{http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd}Id'] = ('Id', 'None', False)
    c_attributes['Usage'] = ('Usage', 'None', False)

    def __init__(self, Id=None, Usage=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.Id = Id
        self.Usage = Usage