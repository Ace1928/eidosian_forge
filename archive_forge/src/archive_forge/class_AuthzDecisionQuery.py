import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class AuthzDecisionQuery(AuthzDecisionQueryType_):
    """The urn:oasis:names:tc:SAML:2.0:protocol:AuthzDecisionQuery element"""
    c_tag = 'AuthzDecisionQuery'
    c_namespace = NAMESPACE
    c_children = AuthzDecisionQueryType_.c_children.copy()
    c_attributes = AuthzDecisionQueryType_.c_attributes.copy()
    c_child_order = AuthzDecisionQueryType_.c_child_order[:]
    c_cardinality = AuthzDecisionQueryType_.c_cardinality.copy()