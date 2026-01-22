import saml2
from saml2 import SamlBase
class PolicyReference(SamlBase):
    """The http://schemas.xmlsoap.org/ws/2004/09/policy:PolicyReference element"""
    c_tag = 'PolicyReference'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_attributes['URI'] = ('uri', 'anyURI', True)
    c_attributes['Digest'] = ('digest', 'base64Binary', False)
    c_attributes['DigestAlgorithm'] = ('digest_algorithm', 'anyURI', False)

    def __init__(self, uri=None, digest=None, digest_algorithm='http://schemas.xmlsoap.org/ws/2004/09/policy/Sha1Exc', text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.uri = uri
        self.digest = digest
        self.digest_algorithm = digest_algorithm