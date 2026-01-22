import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
class AttributeConsumingServiceType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:metadata:AttributeConsumingServiceType
    element"""
    c_tag = 'AttributeConsumingServiceType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}ServiceName'] = ('service_name', [ServiceName])
    c_cardinality['service_name'] = {'min': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}ServiceDescription'] = ('service_description', [ServiceDescription])
    c_cardinality['service_description'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}RequestedAttribute'] = ('requested_attribute', [RequestedAttribute])
    c_cardinality['requested_attribute'] = {'min': 1}
    c_attributes['index'] = ('index', 'unsignedShort', True)
    c_attributes['isDefault'] = ('is_default', 'boolean', False)
    c_child_order.extend(['service_name', 'service_description', 'requested_attribute'])

    def __init__(self, service_name=None, service_description=None, requested_attribute=None, index=None, is_default=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.service_name = service_name or []
        self.service_description = service_description or []
        self.requested_attribute = requested_attribute or []
        self.index = index
        self.is_default = is_default