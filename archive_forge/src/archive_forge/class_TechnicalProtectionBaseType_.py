import saml2
from saml2 import SamlBase
class TechnicalProtectionBaseType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:TechnicalProtectionBaseType element"""
    c_tag = 'TechnicalProtectionBaseType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}PrivateKeyProtection'] = ('private_key_protection', PrivateKeyProtection)
    c_cardinality['private_key_protection'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}SecretKeyProtection'] = ('secret_key_protection', SecretKeyProtection)
    c_cardinality['secret_key_protection'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}Extension'] = ('extension', [Extension])
    c_cardinality['extension'] = {'min': 0}
    c_child_order.extend(['private_key_protection', 'secret_key_protection', 'extension'])

    def __init__(self, private_key_protection=None, secret_key_protection=None, extension=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.private_key_protection = private_key_protection
        self.secret_key_protection = secret_key_protection
        self.extension = extension or []