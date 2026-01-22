import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class ManageNameIDRequestType_(RequestAbstractType_):
    """
    The urn:oasis:names:tc:SAML:2.0:protocol:ManageNameIDRequestType element
    """
    c_tag = 'ManageNameIDRequestType'
    c_namespace = NAMESPACE
    c_children = RequestAbstractType_.c_children.copy()
    c_attributes = RequestAbstractType_.c_attributes.copy()
    c_child_order = RequestAbstractType_.c_child_order[:]
    c_cardinality = RequestAbstractType_.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}NameID'] = ('name_id', saml.NameID)
    c_cardinality['name_id'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}EncryptedID'] = ('encrypted_id', saml.EncryptedID)
    c_cardinality['encrypted_id'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:protocol}NewID'] = ('new_id', NewID)
    c_cardinality['new_id'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:protocol}NewEncryptedID'] = ('new_encrypted_id', NewEncryptedID)
    c_cardinality['new_encrypted_id'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:protocol}Terminate'] = ('terminate', Terminate)
    c_cardinality['terminate'] = {'min': 0, 'max': 1}
    c_child_order.extend(['name_id', 'encrypted_id', 'new_id', 'new_encrypted_id', 'terminate'])

    def __init__(self, name_id=None, encrypted_id=None, new_id=None, new_encrypted_id=None, terminate=None, issuer=None, signature=None, extensions=None, id=None, version=None, issue_instant=None, destination=None, consent=None, text=None, extension_elements=None, extension_attributes=None):
        RequestAbstractType_.__init__(self, issuer=issuer, signature=signature, extensions=extensions, id=id, version=version, issue_instant=issue_instant, destination=destination, consent=consent, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.name_id = name_id
        self.encrypted_id = encrypted_id
        self.new_id = new_id
        self.new_encrypted_id = new_encrypted_id
        self.terminate = terminate