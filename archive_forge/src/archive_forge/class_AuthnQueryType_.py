import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class AuthnQueryType_(SubjectQueryAbstractType_):
    """The urn:oasis:names:tc:SAML:2.0:protocol:AuthnQueryType element"""
    c_tag = 'AuthnQueryType'
    c_namespace = NAMESPACE
    c_children = SubjectQueryAbstractType_.c_children.copy()
    c_attributes = SubjectQueryAbstractType_.c_attributes.copy()
    c_child_order = SubjectQueryAbstractType_.c_child_order[:]
    c_cardinality = SubjectQueryAbstractType_.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:protocol}RequestedAuthnContext'] = ('requested_authn_context', RequestedAuthnContext)
    c_cardinality['requested_authn_context'] = {'min': 0, 'max': 1}
    c_attributes['SessionIndex'] = ('session_index', 'string', False)
    c_child_order.extend(['requested_authn_context'])

    def __init__(self, requested_authn_context=None, session_index=None, subject=None, issuer=None, signature=None, extensions=None, id=None, version=None, issue_instant=None, destination=None, consent=None, text=None, extension_elements=None, extension_attributes=None):
        SubjectQueryAbstractType_.__init__(self, subject=subject, issuer=issuer, signature=signature, extensions=extensions, id=id, version=version, issue_instant=issue_instant, destination=destination, consent=consent, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.requested_authn_context = requested_authn_context
        self.session_index = session_index