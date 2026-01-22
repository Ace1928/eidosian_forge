import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class NameIDMappingResponse(NameIDMappingResponseType_):
    """The urn:oasis:names:tc:SAML:2.0:protocol:NameIDMappingResponse element"""
    c_tag = 'NameIDMappingResponse'
    c_namespace = NAMESPACE
    c_children = NameIDMappingResponseType_.c_children.copy()
    c_attributes = NameIDMappingResponseType_.c_attributes.copy()
    c_child_order = NameIDMappingResponseType_.c_child_order[:]
    c_cardinality = NameIDMappingResponseType_.c_cardinality.copy()