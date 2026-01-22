import saml2
from saml2 import SamlBase
class X509IssuerSerial(X509IssuerSerialType_):
    c_tag = 'X509IssuerSerial'
    c_namespace = NAMESPACE
    c_children = X509IssuerSerialType_.c_children.copy()
    c_attributes = X509IssuerSerialType_.c_attributes.copy()
    c_child_order = X509IssuerSerialType_.c_child_order[:]
    c_cardinality = X509IssuerSerialType_.c_cardinality.copy()