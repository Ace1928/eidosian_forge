import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class StatusCodeType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:protocol:StatusCodeType element"""
    c_tag = 'StatusCodeType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_cardinality['status_code'] = {'min': 0, 'max': 1}
    c_attributes['Value'] = ('value', 'anyURI', True)
    c_child_order.extend(['status_code'])

    def __init__(self, status_code=None, value=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.status_code = status_code
        self.value = value