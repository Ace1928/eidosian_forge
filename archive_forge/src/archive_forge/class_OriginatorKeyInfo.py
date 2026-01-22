import saml2
from saml2 import SamlBase
from saml2 import xmldsig as ds
class OriginatorKeyInfo(ds.KeyInfo):
    c_tag = 'OriginatorKeyInfo'
    c_namespace = NAMESPACE
    c_children = ds.KeyInfo.c_children.copy()
    c_attributes = ds.KeyInfo.c_attributes.copy()
    c_child_order = ds.KeyInfo.c_child_order[:]
    c_cardinality = ds.KeyInfo.c_cardinality.copy()