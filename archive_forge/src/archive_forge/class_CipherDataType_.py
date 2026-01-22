import saml2
from saml2 import SamlBase
from saml2 import xmldsig as ds
class CipherDataType_(SamlBase):
    """The http://www.w3.org/2001/04/xmlenc#:CipherDataType element"""
    c_tag = 'CipherDataType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://www.w3.org/2001/04/xmlenc#}CipherValue'] = ('cipher_value', CipherValue)
    c_cardinality['cipher_value'] = {'min': 0, 'max': 1}
    c_children['{http://www.w3.org/2001/04/xmlenc#}CipherReference'] = ('cipher_reference', CipherReference)
    c_cardinality['cipher_reference'] = {'min': 0, 'max': 1}
    c_child_order.extend(['cipher_value', 'cipher_reference'])

    def __init__(self, cipher_value=None, cipher_reference=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.cipher_value = cipher_value
        self.cipher_reference = cipher_reference