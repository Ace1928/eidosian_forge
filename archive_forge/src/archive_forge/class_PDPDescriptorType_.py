import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
class PDPDescriptorType_(RoleDescriptorType_):
    """The urn:oasis:names:tc:SAML:2.0:metadata:PDPDescriptorType element"""
    c_tag = 'PDPDescriptorType'
    c_namespace = NAMESPACE
    c_children = RoleDescriptorType_.c_children.copy()
    c_attributes = RoleDescriptorType_.c_attributes.copy()
    c_child_order = RoleDescriptorType_.c_child_order[:]
    c_cardinality = RoleDescriptorType_.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}AuthzService'] = ('authz_service', [AuthzService])
    c_cardinality['authz_service'] = {'min': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}AssertionIDRequestService'] = ('assertion_id_request_service', [AssertionIDRequestService])
    c_cardinality['assertion_id_request_service'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}NameIDFormat'] = ('name_id_format', [NameIDFormat])
    c_cardinality['name_id_format'] = {'min': 0}
    c_child_order.extend(['authz_service', 'assertion_id_request_service', 'name_id_format'])

    def __init__(self, authz_service=None, assertion_id_request_service=None, name_id_format=None, signature=None, extensions=None, key_descriptor=None, organization=None, contact_person=None, id=None, valid_until=None, cache_duration=None, protocol_support_enumeration=None, error_url=None, text=None, extension_elements=None, extension_attributes=None):
        RoleDescriptorType_.__init__(self, signature=signature, extensions=extensions, key_descriptor=key_descriptor, organization=organization, contact_person=contact_person, id=id, valid_until=valid_until, cache_duration=cache_duration, protocol_support_enumeration=protocol_support_enumeration, error_url=error_url, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.authz_service = authz_service or []
        self.assertion_id_request_service = assertion_id_request_service or []
        self.name_id_format = name_id_format or []