import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class SignChallengeResponse(SignChallengeType_):
    """The http://docs.oasis-open.org/ws-sx/ws-trust/200512/:SignChallengeResponse element"""
    c_tag = 'SignChallengeResponse'
    c_namespace = NAMESPACE
    c_children = SignChallengeType_.c_children.copy()
    c_attributes = SignChallengeType_.c_attributes.copy()
    c_child_order = SignChallengeType_.c_child_order[:]
    c_cardinality = SignChallengeType_.c_cardinality.copy()