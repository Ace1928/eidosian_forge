import saml2
from saml2 import SamlBase
class PrincipalAuthenticationMechanismType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:PrincipalAuthenticationMechanismType element"""
    c_tag = 'PrincipalAuthenticationMechanismType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}Password'] = ('password', Password)
    c_cardinality['password'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}RestrictedPassword'] = ('restricted_password', RestrictedPassword)
    c_cardinality['restricted_password'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}Token'] = ('token', Token)
    c_cardinality['token'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}Smartcard'] = ('smartcard', Smartcard)
    c_cardinality['smartcard'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}ActivationPin'] = ('activation_pin', ActivationPin)
    c_cardinality['activation_pin'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}Extension'] = ('extension', [Extension])
    c_cardinality['extension'] = {'min': 0}
    c_attributes['preauth'] = ('preauth', 'integer', False)
    c_child_order.extend(['password', 'restricted_password', 'token', 'smartcard', 'activation_pin', 'extension'])

    def __init__(self, password=None, restricted_password=None, token=None, smartcard=None, activation_pin=None, extension=None, preauth=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.password = password
        self.restricted_password = restricted_password
        self.token = token
        self.smartcard = smartcard
        self.activation_pin = activation_pin
        self.extension = extension or []
        self.preauth = preauth