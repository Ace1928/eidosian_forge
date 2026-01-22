import saml2
from saml2 import SamlBase
class AuthnMethodBaseType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:AuthnMethodBaseType element"""
    c_tag = 'AuthnMethodBaseType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}PrincipalAuthenticationMechanism'] = ('principal_authentication_mechanism', PrincipalAuthenticationMechanism)
    c_cardinality['principal_authentication_mechanism'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}Authenticator'] = ('authenticator', Authenticator)
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}AuthenticatorTransportProtocol'] = ('authenticator_transport_protocol', AuthenticatorTransportProtocol)
    c_cardinality['authenticator_transport_protocol'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}Extension'] = ('extension', [Extension])
    c_cardinality['extension'] = {'min': 0}
    c_child_order.extend(['principal_authentication_mechanism', 'authenticator', 'authenticator_transport_protocol', 'extension'])

    def __init__(self, principal_authentication_mechanism=None, authenticator=None, authenticator_transport_protocol=None, extension=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.principal_authentication_mechanism = principal_authentication_mechanism
        self.authenticator = authenticator
        self.authenticator_transport_protocol = authenticator_transport_protocol
        self.extension = extension or []