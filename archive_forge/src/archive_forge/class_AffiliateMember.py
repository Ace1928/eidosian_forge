import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
class AffiliateMember(EntityIDType_):
    """The urn:oasis:names:tc:SAML:2.0:metadata:AffiliateMember element"""
    c_tag = 'AffiliateMember'
    c_namespace = NAMESPACE
    c_children = EntityIDType_.c_children.copy()
    c_attributes = EntityIDType_.c_attributes.copy()
    c_child_order = EntityIDType_.c_child_order[:]
    c_cardinality = EntityIDType_.c_cardinality.copy()