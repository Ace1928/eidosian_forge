import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class ArtifactResolve(ArtifactResolveType_):
    """The urn:oasis:names:tc:SAML:2.0:protocol:ArtifactResolve element"""
    c_tag = 'ArtifactResolve'
    c_namespace = NAMESPACE
    c_children = ArtifactResolveType_.c_children.copy()
    c_attributes = ArtifactResolveType_.c_attributes.copy()
    c_child_order = ArtifactResolveType_.c_child_order[:]
    c_cardinality = ArtifactResolveType_.c_cardinality.copy()