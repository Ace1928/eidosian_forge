import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class ProofEncryption(ProofEncryptionType_):
    """The http://docs.oasis-open.org/ws-sx/ws-trust/200512/:ProofEncryption element"""
    c_tag = 'ProofEncryption'
    c_namespace = NAMESPACE
    c_children = ProofEncryptionType_.c_children.copy()
    c_attributes = ProofEncryptionType_.c_attributes.copy()
    c_child_order = ProofEncryptionType_.c_child_order[:]
    c_cardinality = ProofEncryptionType_.c_cardinality.copy()