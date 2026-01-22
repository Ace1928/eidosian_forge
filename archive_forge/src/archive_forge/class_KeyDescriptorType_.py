import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
class KeyDescriptorType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:metadata:KeyDescriptorType element"""
    c_tag = 'KeyDescriptorType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://www.w3.org/2000/09/xmldsig#}KeyInfo'] = ('key_info', ds.KeyInfo)
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}EncryptionMethod'] = ('encryption_method', [EncryptionMethod])
    c_cardinality['encryption_method'] = {'min': 0}
    c_attributes['use'] = ('use', KeyTypes_, False)
    c_child_order.extend(['key_info', 'encryption_method'])

    def __init__(self, key_info=None, encryption_method=None, use=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.key_info = key_info
        self.encryption_method = encryption_method or []
        self.use = use