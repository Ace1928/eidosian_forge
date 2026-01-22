import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import samlp
class RelayState(RelayStateType_):
    """The urn:oasis:names:tc:SAML:2.0:profiles:SSO:ecp:RelayState element"""
    c_tag = 'RelayState'
    c_namespace = NAMESPACE
    c_children = RelayStateType_.c_children.copy()
    c_attributes = RelayStateType_.c_attributes.copy()
    c_child_order = RelayStateType_.c_child_order[:]
    c_cardinality = RelayStateType_.c_cardinality.copy()