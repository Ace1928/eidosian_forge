import saml2
from saml2 import SamlBase
class ActivationLimit(ActivationLimitType_):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:ActivationLimit element"""
    c_tag = 'ActivationLimit'
    c_namespace = NAMESPACE
    c_children = ActivationLimitType_.c_children.copy()
    c_attributes = ActivationLimitType_.c_attributes.copy()
    c_child_order = ActivationLimitType_.c_child_order[:]
    c_cardinality = ActivationLimitType_.c_cardinality.copy()