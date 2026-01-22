import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class NameIDPolicy(NameIDPolicyType_):
    """The urn:oasis:names:tc:SAML:2.0:protocol:NameIDPolicy element"""
    c_tag = 'NameIDPolicy'
    c_namespace = NAMESPACE
    c_children = NameIDPolicyType_.c_children.copy()
    c_attributes = NameIDPolicyType_.c_attributes.copy()
    c_child_order = NameIDPolicyType_.c_child_order[:]
    c_cardinality = NameIDPolicyType_.c_cardinality.copy()