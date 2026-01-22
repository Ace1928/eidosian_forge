import saml2
from saml2 import SamlBase
from saml2 import saml
class RequestedAttributeType_(SamlBase):
    """The http://eidas.europa.eu/saml-extensions:RequestedAttributeType element"""
    c_tag = 'RequestedAttributeType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}AttributeValue'] = ('attribute_value', [saml.AttributeValue])
    c_cardinality['attribute_value'] = {'min': 0}
    c_attributes['Name'] = ('name', 'None', True)
    c_attributes['NameFormat'] = ('name_format', 'None', True)
    c_attributes['FriendlyName'] = ('friendly_name', 'None', False)
    c_attributes['isRequired'] = ('is_required', 'None', False)
    c_child_order.extend(['attribute_value'])

    def __init__(self, attribute_value=None, name=None, name_format=None, friendly_name=None, is_required=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.attribute_value = attribute_value or []
        self.name = name
        self.name_format = name_format
        self.friendly_name = friendly_name
        self.is_required = is_required