import saml2
from saml2 import md
class RequestInitiator(md.EndpointType_):
    """The urn:oasis:names:tc:SAML:profiles:SSO:request-init:RequestInitiator
    element"""
    c_tag = 'RequestInitiator'
    c_namespace = NAMESPACE
    c_children = md.EndpointType_.c_children.copy()
    c_attributes = md.EndpointType_.c_attributes.copy()
    c_child_order = md.EndpointType_.c_child_order[:]
    c_cardinality = md.EndpointType_.c_cardinality.copy()