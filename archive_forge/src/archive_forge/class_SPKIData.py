import saml2
from saml2 import SamlBase
class SPKIData(SPKIDataType_):
    """The http://www.w3.org/2000/09/xmldsig#:SPKIData element"""
    c_tag = 'SPKIData'
    c_namespace = NAMESPACE
    c_children = SPKIDataType_.c_children.copy()
    c_attributes = SPKIDataType_.c_attributes.copy()
    c_child_order = SPKIDataType_.c_child_order[:]
    c_cardinality = SPKIDataType_.c_cardinality.copy()