import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class Scoping(ScopingType_):
    """The urn:oasis:names:tc:SAML:2.0:protocol:Scoping element"""
    c_tag = 'Scoping'
    c_namespace = NAMESPACE
    c_children = ScopingType_.c_children.copy()
    c_attributes = ScopingType_.c_attributes.copy()
    c_child_order = ScopingType_.c_child_order[:]
    c_cardinality = ScopingType_.c_cardinality.copy()