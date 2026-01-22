import saml2
from saml2 import SamlBase
from saml2 import xmldsig as ds
class EncryptedKeyType_(EncryptedType_):
    """The http://www.w3.org/2001/04/xmlenc#:EncryptedKeyType element"""
    c_tag = 'EncryptedKeyType'
    c_namespace = NAMESPACE
    c_children = EncryptedType_.c_children.copy()
    c_attributes = EncryptedType_.c_attributes.copy()
    c_child_order = EncryptedType_.c_child_order[:]
    c_cardinality = EncryptedType_.c_cardinality.copy()
    c_children['{http://www.w3.org/2001/04/xmlenc#}ReferenceList'] = ('reference_list', ReferenceList)
    c_cardinality['reference_list'] = {'min': 0, 'max': 1}
    c_children['{http://www.w3.org/2001/04/xmlenc#}CarriedKeyName'] = ('carried_key_name', CarriedKeyName)
    c_cardinality['carried_key_name'] = {'min': 0, 'max': 1}
    c_attributes['Recipient'] = ('recipient', 'string', False)
    c_child_order.extend(['reference_list', 'carried_key_name'])

    def __init__(self, reference_list=None, carried_key_name=None, recipient=None, encryption_method=None, key_info=None, cipher_data=None, encryption_properties=None, id=None, type=None, mime_type=None, encoding=None, text=None, extension_elements=None, extension_attributes=None):
        EncryptedType_.__init__(self, encryption_method=encryption_method, key_info=key_info, cipher_data=cipher_data, encryption_properties=encryption_properties, id=id, type=type, mime_type=mime_type, encoding=encoding, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.reference_list = reference_list
        self.carried_key_name = carried_key_name
        self.recipient = recipient