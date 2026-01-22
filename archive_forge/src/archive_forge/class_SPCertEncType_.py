import saml2
from saml2 import SamlBase
from saml2.xmldsig import KeyInfo
class SPCertEncType_(SamlBase):
    """The urn:net:eustix:names:tc:PEFIM:0.0:assertion:SPCertEncType element"""
    c_tag = 'SPCertEncType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://www.w3.org/2000/09/xmldsig#}KeyInfo'] = ('key_info', [KeyInfo])
    c_cardinality['key_info'] = {'min': 1}
    c_attributes['VerifyDepth'] = ('verify_depth', 'unsignedByte', False)
    c_child_order.extend(['key_info'])

    def __init__(self, key_info=None, x509_data=None, verify_depth='1', text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        if key_info:
            self.key_info = key_info
        elif x509_data:
            self.key_info = KeyInfo(x509_data=x509_data)
        else:
            self.key_info = []
        self.verify_depth = verify_depth