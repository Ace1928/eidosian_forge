import saml2
from saml2 import SamlBase
class SignedInfoType_(SamlBase):
    """The http://www.w3.org/2000/09/xmldsig#:SignedInfoType element"""
    c_tag = 'SignedInfoType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://www.w3.org/2000/09/xmldsig#}CanonicalizationMethod'] = ('canonicalization_method', CanonicalizationMethod)
    c_children['{http://www.w3.org/2000/09/xmldsig#}SignatureMethod'] = ('signature_method', SignatureMethod)
    c_children['{http://www.w3.org/2000/09/xmldsig#}Reference'] = ('reference', [Reference])
    c_cardinality['reference'] = {'min': 1}
    c_attributes['Id'] = ('id', 'ID', False)
    c_child_order.extend(['canonicalization_method', 'signature_method', 'reference'])

    def __init__(self, canonicalization_method=None, signature_method=None, reference=None, id=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.canonicalization_method = canonicalization_method
        self.signature_method = signature_method
        self.reference = reference or []
        self.id = id