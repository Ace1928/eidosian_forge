import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class AssertionIDRequest(AssertionIDRequestType_):
    """The urn:oasis:names:tc:SAML:2.0:protocol:AssertionIDRequest element"""
    c_tag = 'AssertionIDRequest'
    c_namespace = NAMESPACE
    c_children = AssertionIDRequestType_.c_children.copy()
    c_attributes = AssertionIDRequestType_.c_attributes.copy()
    c_child_order = AssertionIDRequestType_.c_child_order[:]
    c_cardinality = AssertionIDRequestType_.c_cardinality.copy()