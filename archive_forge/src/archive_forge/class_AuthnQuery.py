import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class AuthnQuery(AuthnQueryType_):
    """The urn:oasis:names:tc:SAML:2.0:protocol:AuthnQuery element"""
    c_tag = 'AuthnQuery'
    c_namespace = NAMESPACE
    c_children = AuthnQueryType_.c_children.copy()
    c_attributes = AuthnQueryType_.c_attributes.copy()
    c_child_order = AuthnQueryType_.c_child_order[:]
    c_cardinality = AuthnQueryType_.c_cardinality.copy()