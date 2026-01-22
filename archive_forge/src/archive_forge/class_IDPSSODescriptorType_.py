import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
class IDPSSODescriptorType_(SSODescriptorType_):
    """The urn:oasis:names:tc:SAML:2.0:metadata:IDPSSODescriptorType element"""
    c_tag = 'IDPSSODescriptorType'
    c_namespace = NAMESPACE
    c_children = SSODescriptorType_.c_children.copy()
    c_attributes = SSODescriptorType_.c_attributes.copy()
    c_child_order = SSODescriptorType_.c_child_order[:]
    c_cardinality = SSODescriptorType_.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}SingleSignOnService'] = ('single_sign_on_service', [SingleSignOnService])
    c_cardinality['single_sign_on_service'] = {'min': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}NameIDMappingService'] = ('name_id_mapping_service', [NameIDMappingService])
    c_cardinality['name_id_mapping_service'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}AssertionIDRequestService'] = ('assertion_id_request_service', [AssertionIDRequestService])
    c_cardinality['assertion_id_request_service'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}AttributeProfile'] = ('attribute_profile', [AttributeProfile])
    c_cardinality['attribute_profile'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}Attribute'] = ('attribute', [saml.Attribute])
    c_cardinality['attribute'] = {'min': 0}
    c_attributes['WantAuthnRequestsSigned'] = ('want_authn_requests_signed', 'boolean', False)
    c_child_order.extend(['single_sign_on_service', 'name_id_mapping_service', 'assertion_id_request_service', 'attribute_profile', 'attribute'])

    def __init__(self, single_sign_on_service=None, name_id_mapping_service=None, assertion_id_request_service=None, attribute_profile=None, attribute=None, want_authn_requests_signed=None, artifact_resolution_service=None, single_logout_service=None, manage_name_id_service=None, name_id_format=None, signature=None, extensions=None, key_descriptor=None, organization=None, contact_person=None, id=None, valid_until=None, cache_duration=None, protocol_support_enumeration=None, error_url=None, text=None, extension_elements=None, extension_attributes=None, want_authn_requests_only_with_valid_cert=None):
        SSODescriptorType_.__init__(self, artifact_resolution_service=artifact_resolution_service, single_logout_service=single_logout_service, manage_name_id_service=manage_name_id_service, name_id_format=name_id_format, signature=signature, extensions=extensions, key_descriptor=key_descriptor, organization=organization, contact_person=contact_person, id=id, valid_until=valid_until, cache_duration=cache_duration, protocol_support_enumeration=protocol_support_enumeration, error_url=error_url, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.single_sign_on_service = single_sign_on_service or []
        self.name_id_mapping_service = name_id_mapping_service or []
        self.assertion_id_request_service = assertion_id_request_service or []
        self.attribute_profile = attribute_profile or []
        self.attribute = attribute or []
        self.want_authn_requests_signed = want_authn_requests_signed
        self.want_authn_requests_only_with_valid_cert = want_authn_requests_only_with_valid_cert