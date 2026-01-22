import saml2
from saml2 import SamlBase
from saml2 import md
class Keywords(KeywordsType_):
    """The urn:oasis:names:tc:SAML:metadata:ui:Keywords element"""
    c_tag = 'Keywords'
    c_namespace = NAMESPACE
    c_children = KeywordsType_.c_children.copy()
    c_attributes = KeywordsType_.c_attributes.copy()
    c_child_order = KeywordsType_.c_child_order[:]
    c_cardinality = KeywordsType_.c_cardinality.copy()