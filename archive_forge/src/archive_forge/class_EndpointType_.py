import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
class EndpointType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:metadata:EndpointType element"""
    c_tag = 'EndpointType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_attributes['Binding'] = ('binding', 'anyURI', True)
    c_attributes['Location'] = ('location', 'anyURI', True)
    c_attributes['ResponseLocation'] = ('response_location', 'anyURI', False)

    def __init__(self, binding=None, location=None, response_location=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.binding = binding
        self.location = location
        self.response_location = response_location