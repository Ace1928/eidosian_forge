import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
class NameIDMappingService(EndpointType_):
    """The urn:oasis:names:tc:SAML:2.0:metadata:NameIDMappingService element"""
    c_tag = 'NameIDMappingService'
    c_namespace = NAMESPACE
    c_children = EndpointType_.c_children.copy()
    c_attributes = EndpointType_.c_attributes.copy()
    c_child_order = EndpointType_.c_child_order[:]
    c_cardinality = EndpointType_.c_cardinality.copy()