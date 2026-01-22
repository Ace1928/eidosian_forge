import saml2
from saml2 import SamlBase
class RSAKeyValueType_(SamlBase):
    """The http://www.w3.org/2000/09/xmldsig#:RSAKeyValueType element"""
    c_tag = 'RSAKeyValueType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://www.w3.org/2000/09/xmldsig#}Modulus'] = ('modulus', Modulus)
    c_children['{http://www.w3.org/2000/09/xmldsig#}Exponent'] = ('exponent', Exponent)
    c_child_order.extend(['modulus', 'exponent'])

    def __init__(self, modulus=None, exponent=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.modulus = modulus
        self.exponent = exponent