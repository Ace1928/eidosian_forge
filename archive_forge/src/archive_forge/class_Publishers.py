import saml2
from saml2 import SamlBase
from saml2 import md
class Publishers(PublishersType_):
    """The urn:oasis:names:tc:SAML:2.0:metadata:dri:Publishers element"""
    c_tag = 'Publishers'
    c_namespace = NAMESPACE
    c_children = PublishersType_.c_children.copy()
    c_attributes = PublishersType_.c_attributes.copy()
    c_child_order = PublishersType_.c_child_order[:]
    c_cardinality = PublishersType_.c_cardinality.copy()