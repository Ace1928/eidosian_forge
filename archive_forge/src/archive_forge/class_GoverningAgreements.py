import saml2
from saml2 import SamlBase
class GoverningAgreements(GoverningAgreementsType_):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:GoverningAgreements element"""
    c_tag = 'GoverningAgreements'
    c_namespace = NAMESPACE
    c_children = GoverningAgreementsType_.c_children.copy()
    c_attributes = GoverningAgreementsType_.c_attributes.copy()
    c_child_order = GoverningAgreementsType_.c_child_order[:]
    c_cardinality = GoverningAgreementsType_.c_cardinality.copy()