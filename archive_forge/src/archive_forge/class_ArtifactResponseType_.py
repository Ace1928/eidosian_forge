import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class ArtifactResponseType_(StatusResponseType_):
    """The urn:oasis:names:tc:SAML:2.0:protocol:ArtifactResponseType element"""
    c_tag = 'ArtifactResponseType'
    c_namespace = NAMESPACE
    c_children = StatusResponseType_.c_children.copy()
    c_attributes = StatusResponseType_.c_attributes.copy()
    c_child_order = StatusResponseType_.c_child_order[:]
    c_cardinality = StatusResponseType_.c_cardinality.copy()
    c_any = {'namespace': '##any', 'processContents': 'lax', 'minOccurs': '0'}