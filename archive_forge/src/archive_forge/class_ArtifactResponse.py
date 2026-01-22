import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class ArtifactResponse(ArtifactResponseType_):
    """The urn:oasis:names:tc:SAML:2.0:protocol:ArtifactResponse element"""
    c_tag = 'ArtifactResponse'
    c_namespace = NAMESPACE
    c_children = ArtifactResponseType_.c_children.copy()
    c_attributes = ArtifactResponseType_.c_attributes.copy()
    c_child_order = ArtifactResponseType_.c_child_order[:]
    c_cardinality = ArtifactResponseType_.c_cardinality.copy()