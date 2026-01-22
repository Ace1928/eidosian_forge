import saml2
from saml2 import SamlBase
class ComplexAuthenticatorType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:ComplexAuthenticatorType element"""
    c_tag = 'ComplexAuthenticatorType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}PreviousSession'] = ('previous_session', PreviousSession)
    c_cardinality['previous_session'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}ResumeSession'] = ('resume_session', ResumeSession)
    c_cardinality['resume_session'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}DigSig'] = ('dig_sig', DigSig)
    c_cardinality['dig_sig'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}Password'] = ('password', Password)
    c_cardinality['password'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}RestrictedPassword'] = ('restricted_password', RestrictedPassword)
    c_cardinality['restricted_password'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}ZeroKnowledge'] = ('zero_knowledge', ZeroKnowledge)
    c_cardinality['zero_knowledge'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}SharedSecretChallengeResponse'] = ('shared_secret_challenge_response', SharedSecretChallengeResponse)
    c_cardinality['shared_secret_challenge_response'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}SharedSecretDynamicPlaintext'] = ('shared_secret_dynamic_plaintext', SharedSecretDynamicPlaintext)
    c_cardinality['shared_secret_dynamic_plaintext'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}IPAddress'] = ('ip_address', IPAddress)
    c_cardinality['ip_address'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}AsymmetricDecryption'] = ('asymmetric_decryption', AsymmetricDecryption)
    c_cardinality['asymmetric_decryption'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}AsymmetricKeyAgreement'] = ('asymmetric_key_agreement', AsymmetricKeyAgreement)
    c_cardinality['asymmetric_key_agreement'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}SubscriberLineNumber'] = ('subscriber_line_number', SubscriberLineNumber)
    c_cardinality['subscriber_line_number'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}UserSuffix'] = ('user_suffix', UserSuffix)
    c_cardinality['user_suffix'] = {'min': 0, 'max': 1}
    c_cardinality['complex_authenticator'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}Extension'] = ('extension', [Extension])
    c_cardinality['extension'] = {'min': 0}
    c_child_order.extend(['previous_session', 'resume_session', 'dig_sig', 'password', 'restricted_password', 'zero_knowledge', 'shared_secret_challenge_response', 'shared_secret_dynamic_plaintext', 'ip_address', 'asymmetric_decryption', 'asymmetric_key_agreement', 'subscriber_line_number', 'user_suffix', 'complex_authenticator', 'extension'])

    def __init__(self, previous_session=None, resume_session=None, dig_sig=None, password=None, restricted_password=None, zero_knowledge=None, shared_secret_challenge_response=None, shared_secret_dynamic_plaintext=None, ip_address=None, asymmetric_decryption=None, asymmetric_key_agreement=None, subscriber_line_number=None, user_suffix=None, complex_authenticator=None, extension=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.previous_session = previous_session
        self.resume_session = resume_session
        self.dig_sig = dig_sig
        self.password = password
        self.restricted_password = restricted_password
        self.zero_knowledge = zero_knowledge
        self.shared_secret_challenge_response = shared_secret_challenge_response
        self.shared_secret_dynamic_plaintext = shared_secret_dynamic_plaintext
        self.ip_address = ip_address
        self.asymmetric_decryption = asymmetric_decryption
        self.asymmetric_key_agreement = asymmetric_key_agreement
        self.subscriber_line_number = subscriber_line_number
        self.user_suffix = user_suffix
        self.complex_authenticator = complex_authenticator
        self.extension = extension or []