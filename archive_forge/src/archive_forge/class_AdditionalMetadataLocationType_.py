import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
class AdditionalMetadataLocationType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:metadata:AdditionalMetadataLocationType
    element"""
    c_tag = 'AdditionalMetadataLocationType'
    c_namespace = NAMESPACE
    c_value_type = {'base': 'anyURI'}
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_attributes['namespace'] = ('namespace', 'anyURI', True)

    def __init__(self, namespace=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.namespace = namespace