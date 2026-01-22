import base64
from datetime import date
from datetime import datetime
import saml2
from saml2 import SamlBase
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
from saml2.validate import MustValueError
from saml2.validate import ShouldValueError
from saml2.validate import valid_domain_name
from saml2.validate import valid_ipv4
from saml2.validate import valid_ipv6
class EncryptedElementType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:assertion:EncryptedElementType element"""
    c_tag = 'EncryptedElementType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://www.w3.org/2001/04/xmlenc#}EncryptedData'] = ('encrypted_data', xenc.EncryptedData)
    c_children['{http://www.w3.org/2001/04/xmlenc#}EncryptedKey'] = ('encrypted_key', [xenc.EncryptedKey])
    c_cardinality['encrypted_key'] = {'min': 0}
    c_child_order.extend(['encrypted_data', 'encrypted_key'])

    def __init__(self, encrypted_data=None, encrypted_key=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.encrypted_data = encrypted_data
        self.encrypted_key = encrypted_key or []