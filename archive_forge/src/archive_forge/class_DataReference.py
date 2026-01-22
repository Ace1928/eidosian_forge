import saml2
from saml2 import SamlBase
from saml2 import xmldsig as ds
class DataReference(ReferenceType_):
    c_tag = 'DataReference'
    c_namespace = NAMESPACE
    c_children = ReferenceType_.c_children.copy()
    c_attributes = ReferenceType_.c_attributes.copy()
    c_child_order = ReferenceType_.c_child_order[:]
    c_cardinality = ReferenceType_.c_cardinality.copy()