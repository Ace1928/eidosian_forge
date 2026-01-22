import saml2
from saml2 import SamlBase
from saml2 import md
class UIInfo(UIInfoType_):
    """The urn:oasis:names:tc:SAML:metadata:ui:UIInfo element"""
    c_tag = 'UIInfo'
    c_namespace = NAMESPACE
    c_children = UIInfoType_.c_children.copy()
    c_attributes = UIInfoType_.c_attributes.copy()
    c_child_order = UIInfoType_.c_child_order[:]
    c_cardinality = UIInfoType_.c_cardinality.copy()