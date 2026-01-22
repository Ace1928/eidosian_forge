import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
class OrganizationType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:metadata:OrganizationType element"""
    c_tag = 'OrganizationType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}Extensions'] = ('extensions', Extensions)
    c_cardinality['extensions'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}OrganizationName'] = ('organization_name', [OrganizationName])
    c_cardinality['organization_name'] = {'min': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}OrganizationDisplayName'] = ('organization_display_name', [OrganizationDisplayName])
    c_cardinality['organization_display_name'] = {'min': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}OrganizationURL'] = ('organization_url', [OrganizationURL])
    c_cardinality['organization_url'] = {'min': 1}
    c_child_order.extend(['extensions', 'organization_name', 'organization_display_name', 'organization_url'])

    def __init__(self, extensions=None, organization_name=None, organization_display_name=None, organization_url=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.extensions = extensions
        self.organization_name = organization_name or []
        self.organization_display_name = organization_display_name or []
        self.organization_url = organization_url or []