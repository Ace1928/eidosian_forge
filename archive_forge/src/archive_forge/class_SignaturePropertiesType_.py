import saml2
from saml2 import SamlBase
class SignaturePropertiesType_(SamlBase):
    """The http://www.w3.org/2000/09/xmldsig#:SignaturePropertiesType element"""
    c_tag = 'SignaturePropertiesType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://www.w3.org/2000/09/xmldsig#}SignatureProperty'] = ('signature_property', [SignatureProperty])
    c_cardinality['signature_property'] = {'min': 1}
    c_attributes['Id'] = ('id', 'ID', False)
    c_child_order.extend(['signature_property'])

    def __init__(self, signature_property=None, id=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.signature_property = signature_property or []
        self.id = id