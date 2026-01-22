import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
class RoleDescriptorType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:metadata:RoleDescriptorType element"""
    c_tag = 'RoleDescriptorType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://www.w3.org/2000/09/xmldsig#}Signature'] = ('signature', ds.Signature)
    c_cardinality['signature'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}Extensions'] = ('extensions', Extensions)
    c_cardinality['extensions'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}KeyDescriptor'] = ('key_descriptor', [KeyDescriptor])
    c_cardinality['key_descriptor'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}Organization'] = ('organization', Organization)
    c_cardinality['organization'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}ContactPerson'] = ('contact_person', [ContactPerson])
    c_cardinality['contact_person'] = {'min': 0}
    c_attributes['ID'] = ('id', 'ID', False)
    c_attributes['validUntil'] = ('valid_until', 'dateTime', False)
    c_attributes['cacheDuration'] = ('cache_duration', 'duration', False)
    c_attributes['protocolSupportEnumeration'] = ('protocol_support_enumeration', AnyURIListType_, True)
    c_attributes['errorURL'] = ('error_url', 'anyURI', False)
    c_child_order.extend(['signature', 'extensions', 'key_descriptor', 'organization', 'contact_person'])

    def __init__(self, signature=None, extensions=None, key_descriptor=None, organization=None, contact_person=None, id=None, valid_until=None, cache_duration=None, protocol_support_enumeration=None, error_url=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.signature = signature
        self.extensions = extensions
        self.key_descriptor = key_descriptor or []
        self.organization = organization
        self.contact_person = contact_person or []
        self.id = id
        self.valid_until = valid_until
        self.cache_duration = cache_duration
        self.protocol_support_enumeration = protocol_support_enumeration
        self.error_url = error_url