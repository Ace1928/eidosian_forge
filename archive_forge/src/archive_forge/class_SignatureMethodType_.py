import saml2
from saml2 import SamlBase
class SignatureMethodType_(SamlBase):
    """The http://www.w3.org/2000/09/xmldsig#:SignatureMethodType element"""
    c_tag = 'SignatureMethodType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://www.w3.org/2000/09/xmldsig#}HMACOutputLength'] = ('hmac_output_length', HMACOutputLength)
    c_cardinality['hmac_output_length'] = {'min': 0, 'max': 1}
    c_attributes['Algorithm'] = ('algorithm', 'anyURI', True)
    c_child_order.extend(['hmac_output_length'])

    def __init__(self, hmac_output_length=None, algorithm=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.hmac_output_length = hmac_output_length
        self.algorithm = algorithm