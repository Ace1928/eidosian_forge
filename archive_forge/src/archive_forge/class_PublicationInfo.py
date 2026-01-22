import saml2
from saml2 import SamlBase
from saml2 import md
class PublicationInfo(PublicationInfoType_):
    """The urn:oasis:names:tc:SAML:metadata:rpi:PublicationInfo element"""
    c_tag = 'PublicationInfo'
    c_namespace = NAMESPACE
    c_children = PublicationInfoType_.c_children.copy()
    c_attributes = PublicationInfoType_.c_attributes.copy()
    c_child_order = PublicationInfoType_.c_child_order[:]
    c_cardinality = PublicationInfoType_.c_cardinality.copy()