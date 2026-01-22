import saml2
from saml2 import SamlBase
class SharedSecretChallengeResponse(SharedSecretChallengeResponseType_):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:SharedSecretChallengeResponse element"""
    c_tag = 'SharedSecretChallengeResponse'
    c_namespace = NAMESPACE
    c_children = SharedSecretChallengeResponseType_.c_children.copy()
    c_attributes = SharedSecretChallengeResponseType_.c_attributes.copy()
    c_child_order = SharedSecretChallengeResponseType_.c_child_order[:]
    c_cardinality = SharedSecretChallengeResponseType_.c_cardinality.copy()