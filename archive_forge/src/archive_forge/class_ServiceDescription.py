import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
class ServiceDescription(LocalizedNameType_):
    """The urn:oasis:names:tc:SAML:2.0:metadata:ServiceDescription element"""
    c_tag = 'ServiceDescription'
    c_namespace = NAMESPACE
    c_children = LocalizedNameType_.c_children.copy()
    c_attributes = LocalizedNameType_.c_attributes.copy()
    c_child_order = LocalizedNameType_.c_child_order[:]
    c_cardinality = LocalizedNameType_.c_cardinality.copy()