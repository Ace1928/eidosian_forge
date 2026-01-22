import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class AttributeQuery(AttributeQueryType_):
    """The urn:oasis:names:tc:SAML:2.0:protocol:AttributeQuery element"""
    c_tag = 'AttributeQuery'
    c_namespace = NAMESPACE
    c_children = AttributeQueryType_.c_children.copy()
    c_attributes = AttributeQueryType_.c_attributes.copy()
    c_child_order = AttributeQueryType_.c_child_order[:]
    c_cardinality = AttributeQueryType_.c_cardinality.copy()