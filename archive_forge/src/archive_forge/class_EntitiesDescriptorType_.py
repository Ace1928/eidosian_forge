import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
class EntitiesDescriptorType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:metadata:EntitiesDescriptorType
    element"""
    c_tag = 'EntitiesDescriptorType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://www.w3.org/2000/09/xmldsig#}Signature'] = ('signature', ds.Signature)
    c_cardinality['signature'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}Extensions'] = ('extensions', Extensions)
    c_cardinality['extensions'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}EntityDescriptor'] = ('entity_descriptor', [EntityDescriptor])
    c_cardinality['entity_descriptor'] = {'min': 0}
    c_cardinality['entities_descriptor'] = {'min': 0}
    c_attributes['validUntil'] = ('valid_until', 'dateTime', False)
    c_attributes['cacheDuration'] = ('cache_duration', 'duration', False)
    c_attributes['ID'] = ('id', 'ID', False)
    c_attributes['Name'] = ('name', 'string', False)
    c_child_order.extend(['signature', 'extensions', 'entity_descriptor', 'entities_descriptor'])

    def __init__(self, signature=None, extensions=None, entity_descriptor=None, entities_descriptor=None, valid_until=None, cache_duration=None, id=None, name=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.signature = signature
        self.extensions = extensions
        self.entity_descriptor = entity_descriptor or []
        self.entities_descriptor = entities_descriptor or []
        self.valid_until = valid_until
        self.cache_duration = cache_duration
        self.id = id
        self.name = name