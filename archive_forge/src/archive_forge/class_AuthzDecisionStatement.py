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
class AuthzDecisionStatement(AuthzDecisionStatementType_):
    """The urn:oasis:names:tc:SAML:2.0:assertion:AuthzDecisionStatement
    element"""
    c_tag = 'AuthzDecisionStatement'
    c_namespace = NAMESPACE
    c_children = AuthzDecisionStatementType_.c_children.copy()
    c_attributes = AuthzDecisionStatementType_.c_attributes.copy()
    c_child_order = AuthzDecisionStatementType_.c_child_order[:]
    c_cardinality = AuthzDecisionStatementType_.c_cardinality.copy()