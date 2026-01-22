import saml2
from saml2 import SamlBase
class Identification(IdentificationType_):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:Identification element"""
    c_tag = 'Identification'
    c_namespace = NAMESPACE
    c_children = IdentificationType_.c_children.copy()
    c_attributes = IdentificationType_.c_attributes.copy()
    c_child_order = IdentificationType_.c_child_order[:]
    c_cardinality = IdentificationType_.c_cardinality.copy()