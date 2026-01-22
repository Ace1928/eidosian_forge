import saml2
from saml2 import SamlBase
from saml2 import xmldsig as ds
class EncryptionPropertiesType_(SamlBase):
    """The http://www.w3.org/2001/04/xmlenc#:EncryptionPropertiesType element"""
    c_tag = 'EncryptionPropertiesType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://www.w3.org/2001/04/xmlenc#}EncryptionProperty'] = ('encryption_property', [EncryptionProperty])
    c_cardinality['encryption_property'] = {'min': 1}
    c_attributes['Id'] = ('id', 'ID', False)
    c_child_order.extend(['encryption_property'])

    def __init__(self, encryption_property=None, id=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.encryption_property = encryption_property or []
        self.id = id