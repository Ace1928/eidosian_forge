import saml2
from saml2 import SamlBase
class ActivationLimitUsages(ActivationLimitUsagesType_):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:ActivationLimitUsages element"""
    c_tag = 'ActivationLimitUsages'
    c_namespace = NAMESPACE
    c_children = ActivationLimitUsagesType_.c_children.copy()
    c_attributes = ActivationLimitUsagesType_.c_attributes.copy()
    c_child_order = ActivationLimitUsagesType_.c_child_order[:]
    c_cardinality = ActivationLimitUsagesType_.c_cardinality.copy()