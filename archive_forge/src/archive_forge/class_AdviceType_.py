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
class AdviceType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:assertion:AdviceType element"""
    c_tag = 'AdviceType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}AssertionIDRef'] = ('assertion_id_ref', [AssertionIDRef])
    c_cardinality['assertion_id_ref'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}AssertionURIRef'] = ('assertion_uri_ref', [AssertionURIRef])
    c_cardinality['assertion_uri_ref'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}Assertion'] = ('assertion', [Assertion])
    c_cardinality['assertion'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}EncryptedAssertion'] = ('encrypted_assertion', [EncryptedAssertion])
    c_cardinality['encrypted_assertion'] = {'min': 0}
    c_child_order.extend(['assertion_id_ref', 'assertion_uri_ref', 'assertion', 'encrypted_assertion'])
    c_any = {'namespace': '##other', 'processContents': 'lax'}

    def __init__(self, assertion_id_ref=None, assertion_uri_ref=None, assertion=None, encrypted_assertion=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.assertion_id_ref = assertion_id_ref or []
        self.assertion_uri_ref = assertion_uri_ref or []
        self.assertion = assertion or []
        self.encrypted_assertion = encrypted_assertion or []