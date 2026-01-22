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
class AuthzDecisionStatementType_(StatementAbstractType_):
    """The urn:oasis:names:tc:SAML:2.0:assertion:AuthzDecisionStatementType
    element"""
    c_tag = 'AuthzDecisionStatementType'
    c_namespace = NAMESPACE
    c_children = StatementAbstractType_.c_children.copy()
    c_attributes = StatementAbstractType_.c_attributes.copy()
    c_child_order = StatementAbstractType_.c_child_order[:]
    c_cardinality = StatementAbstractType_.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}Action'] = ('action', [Action])
    c_cardinality['action'] = {'min': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}Evidence'] = ('evidence', Evidence)
    c_cardinality['evidence'] = {'min': 0, 'max': 1}
    c_attributes['Resource'] = ('resource', 'anyURI', True)
    c_attributes['Decision'] = ('decision', DecisionType_, True)
    c_child_order.extend(['action', 'evidence'])

    def __init__(self, action=None, evidence=None, resource=None, decision=None, text=None, extension_elements=None, extension_attributes=None):
        StatementAbstractType_.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.action = action or []
        self.evidence = evidence
        self.resource = resource
        self.decision = decision