import saml2
from saml2 import SamlBase
class ReferenceParameters(ReferenceParametersType_):
    """The http://www.w3.org/2005/08/addressing:ReferenceParameters element"""
    c_tag = 'ReferenceParameters'
    c_namespace = NAMESPACE
    c_children = ReferenceParametersType_.c_children.copy()
    c_attributes = ReferenceParametersType_.c_attributes.copy()
    c_child_order = ReferenceParametersType_.c_child_order[:]
    c_cardinality = ReferenceParametersType_.c_cardinality.copy()