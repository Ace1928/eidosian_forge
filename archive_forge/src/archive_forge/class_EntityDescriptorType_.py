import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
class EntityDescriptorType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:metadata:EntityDescriptorType element"""
    c_tag = 'EntityDescriptorType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://www.w3.org/2000/09/xmldsig#}Signature'] = ('signature', ds.Signature)
    c_cardinality['signature'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}Extensions'] = ('extensions', Extensions)
    c_cardinality['extensions'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}RoleDescriptor'] = ('role_descriptor', [RoleDescriptor])
    c_cardinality['role_descriptor'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}IDPSSODescriptor'] = ('idpsso_descriptor', [IDPSSODescriptor])
    c_cardinality['idpsso_descriptor'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}SPSSODescriptor'] = ('spsso_descriptor', [SPSSODescriptor])
    c_cardinality['spsso_descriptor'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}AuthnAuthorityDescriptor'] = ('authn_authority_descriptor', [AuthnAuthorityDescriptor])
    c_cardinality['authn_authority_descriptor'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}AttributeAuthorityDescriptor'] = ('attribute_authority_descriptor', [AttributeAuthorityDescriptor])
    c_cardinality['attribute_authority_descriptor'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}PDPDescriptor'] = ('pdp_descriptor', [PDPDescriptor])
    c_cardinality['pdp_descriptor'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}AffiliationDescriptor'] = ('affiliation_descriptor', AffiliationDescriptor)
    c_cardinality['affiliation_descriptor'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}Organization'] = ('organization', Organization)
    c_cardinality['organization'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}ContactPerson'] = ('contact_person', [ContactPerson])
    c_cardinality['contact_person'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}AdditionalMetadataLocation'] = ('additional_metadata_location', [AdditionalMetadataLocation])
    c_cardinality['additional_metadata_location'] = {'min': 0}
    c_attributes['entityID'] = ('entity_id', EntityIDType_, True)
    c_attributes['validUntil'] = ('valid_until', 'dateTime', False)
    c_attributes['cacheDuration'] = ('cache_duration', 'duration', False)
    c_attributes['ID'] = ('id', 'ID', False)
    c_child_order.extend(['signature', 'extensions', 'role_descriptor', 'idpsso_descriptor', 'spsso_descriptor', 'authn_authority_descriptor', 'attribute_authority_descriptor', 'pdp_descriptor', 'affiliation_descriptor', 'organization', 'contact_person', 'additional_metadata_location'])

    def __init__(self, signature=None, extensions=None, role_descriptor=None, idpsso_descriptor=None, spsso_descriptor=None, authn_authority_descriptor=None, attribute_authority_descriptor=None, pdp_descriptor=None, affiliation_descriptor=None, organization=None, contact_person=None, additional_metadata_location=None, entity_id=None, valid_until=None, cache_duration=None, id=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.signature = signature
        self.extensions = extensions
        self.role_descriptor = role_descriptor or []
        self.idpsso_descriptor = idpsso_descriptor or []
        self.spsso_descriptor = spsso_descriptor or []
        self.authn_authority_descriptor = authn_authority_descriptor or []
        self.attribute_authority_descriptor = attribute_authority_descriptor or []
        self.pdp_descriptor = pdp_descriptor or []
        self.affiliation_descriptor = affiliation_descriptor
        self.organization = organization
        self.contact_person = contact_person or []
        self.additional_metadata_location = additional_metadata_location or []
        self.entity_id = entity_id
        self.valid_until = valid_until
        self.cache_duration = cache_duration
        self.id = id