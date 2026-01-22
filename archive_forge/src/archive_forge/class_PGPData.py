import saml2
from saml2 import SamlBase
class PGPData(PGPDataType_):
    """The http://www.w3.org/2000/09/xmldsig#:PGPData element"""
    c_tag = 'PGPData'
    c_namespace = NAMESPACE
    c_children = PGPDataType_.c_children.copy()
    c_attributes = PGPDataType_.c_attributes.copy()
    c_child_order = PGPDataType_.c_child_order[:]
    c_cardinality = PGPDataType_.c_cardinality.copy()