import base64
from datetime import date
from datetime import datetime
import saml2
from saml2 import SamlBase
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
from saml2.validate import MustValueError
from saml2.validate import ShouldValueError
from saml2.validate import valid_domain_name
from saml2.validate import valid_ipv4
from saml2.validate import valid_ipv6
class AuthnContextType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:assertion:AuthnContextType element"""
    c_tag = 'AuthnContextType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}AuthnContextClassRef'] = ('authn_context_class_ref', AuthnContextClassRef)
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}AuthnContextDecl'] = ('authn_context_decl', AuthnContextDecl)
    c_cardinality['authn_context_decl'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}AuthnContextDeclRef'] = ('authn_context_decl_ref', AuthnContextDeclRef)
    c_cardinality['authn_context_decl_ref'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}AuthenticatingAuthority'] = ('authenticating_authority', [AuthenticatingAuthority])
    c_cardinality['authenticating_authority'] = {'min': 0}
    c_child_order.extend(['authn_context_class_ref', 'authn_context_decl', 'authn_context_decl_ref', 'authenticating_authority'])

    def __init__(self, authn_context_class_ref=None, authn_context_decl=None, authn_context_decl_ref=None, authenticating_authority=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.authn_context_class_ref = authn_context_class_ref
        self.authn_context_decl = authn_context_decl
        self.authn_context_decl_ref = authn_context_decl_ref
        self.authenticating_authority = authenticating_authority or []

    def verify(self):
        if self.authn_context_decl and self.authn_context_decl_ref:
            raise Exception('Invalid Response: Cannot have both <AuthnContextDecl> and <AuthnContextDeclRef>')
        return SamlBase.verify(self)