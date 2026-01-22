import saml2
from saml2 import SamlBase
class SigningMethodType_(SamlBase):
    """The urn:oasis:names:tc:SAML:metadata:algsupport:SigningMethodType
    element"""
    c_tag = 'SigningMethodType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_attributes['Algorithm'] = ('algorithm', 'anyURI', True)
    c_attributes['MinKeySize'] = ('min_key_size', 'positiveInteger', False)
    c_attributes['MaxKeySize'] = ('max_key_size', 'positiveInteger', False)

    def __init__(self, algorithm=None, min_key_size=None, max_key_size=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.algorithm = algorithm
        self.min_key_size = min_key_size
        self.max_key_size = max_key_size