from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
from heat.engine import translation
class IPsecPolicy(neutron.NeutronResource):
    """A resource for IPsec policy in Neutron.

    The IP security policy specifying the authentication and encryption
    algorithm, and encapsulation mode used for the established VPN connection.
    """
    required_service_extension = 'vpnaas'
    entity = 'ipsecpolicy'
    PROPERTIES = NAME, DESCRIPTION, TRANSFORM_PROTOCOL, ENCAPSULATION_MODE, AUTH_ALGORITHM, ENCRYPTION_ALGORITHM, LIFETIME, PFS = ('name', 'description', 'transform_protocol', 'encapsulation_mode', 'auth_algorithm', 'encryption_algorithm', 'lifetime', 'pfs')
    _LIFETIME_KEYS = LIFETIME_UNITS, LIFETIME_VALUE = ('units', 'value')
    ATTRIBUTES = AUTH_ALGORITHM_ATTR, DESCRIPTION_ATTR, ENCAPSULATION_MODE_ATTR, ENCRYPTION_ALGORITHM_ATTR, LIFETIME_ATTR, NAME_ATTR, PFS_ATTR, TENANT_ID, TRANSFORM_PROTOCOL_ATTR = ('auth_algorithm', 'description', 'encapsulation_mode', 'encryption_algorithm', 'lifetime', 'name', 'pfs', 'tenant_id', 'transform_protocol')
    properties_schema = {NAME: properties.Schema(properties.Schema.STRING, _('Name for the ipsec policy.'), update_allowed=True), DESCRIPTION: properties.Schema(properties.Schema.STRING, _('Description for the ipsec policy.'), update_allowed=True), TRANSFORM_PROTOCOL: properties.Schema(properties.Schema.STRING, _('Transform protocol for the ipsec policy.'), default='esp', constraints=[constraints.AllowedValues(['esp', 'ah', 'ah-esp'])]), ENCAPSULATION_MODE: properties.Schema(properties.Schema.STRING, _('Encapsulation mode for the ipsec policy.'), default='tunnel', constraints=[constraints.AllowedValues(['tunnel', 'transport'])]), AUTH_ALGORITHM: properties.Schema(properties.Schema.STRING, _('Authentication hash algorithm for the ipsec policy.'), default='sha1', constraints=[constraints.AllowedValues(['sha1'])]), ENCRYPTION_ALGORITHM: properties.Schema(properties.Schema.STRING, _('Encryption algorithm for the ipsec policy.'), default='aes-128', constraints=[constraints.AllowedValues(['3des', 'aes-128', 'aes-192', 'aes-256'])]), LIFETIME: properties.Schema(properties.Schema.MAP, _('Safety assessment lifetime configuration for the ipsec policy.'), schema={LIFETIME_UNITS: properties.Schema(properties.Schema.STRING, _('Safety assessment lifetime units.'), default='seconds', constraints=[constraints.AllowedValues(['seconds', 'kilobytes'])]), LIFETIME_VALUE: properties.Schema(properties.Schema.INTEGER, _('Safety assessment lifetime value in specified units.'), default=3600)}), PFS: properties.Schema(properties.Schema.STRING, _('Perfect forward secrecy for the ipsec policy.'), default='group5', constraints=[constraints.AllowedValues(['group2', 'group5', 'group14'])])}
    attributes_schema = {AUTH_ALGORITHM_ATTR: attributes.Schema(_('The authentication hash algorithm of the ipsec policy.'), type=attributes.Schema.STRING), DESCRIPTION_ATTR: attributes.Schema(_('The description of the ipsec policy.'), type=attributes.Schema.STRING), ENCAPSULATION_MODE_ATTR: attributes.Schema(_('The encapsulation mode of the ipsec policy.'), type=attributes.Schema.STRING), ENCRYPTION_ALGORITHM_ATTR: attributes.Schema(_('The encryption algorithm of the ipsec policy.'), type=attributes.Schema.STRING), LIFETIME_ATTR: attributes.Schema(_('The safety assessment lifetime configuration of the ipsec policy.'), type=attributes.Schema.MAP), NAME_ATTR: attributes.Schema(_('The name of the ipsec policy.'), type=attributes.Schema.STRING), PFS_ATTR: attributes.Schema(_('The perfect forward secrecy of the ipsec policy.'), type=attributes.Schema.STRING), TENANT_ID: attributes.Schema(_('The unique identifier of the tenant owning the ipsec policy.'), type=attributes.Schema.STRING), TRANSFORM_PROTOCOL_ATTR: attributes.Schema(_('The transform protocol of the ipsec policy.'), type=attributes.Schema.STRING)}

    def handle_create(self):
        props = self.prepare_properties(self.properties, self.physical_resource_name())
        ipsecpolicy = self.client().create_ipsecpolicy({'ipsecpolicy': props})['ipsecpolicy']
        self.resource_id_set(ipsecpolicy['id'])

    def handle_update(self, json_snippet, tmpl_diff, prop_diff):
        if prop_diff:
            self.client().update_ipsecpolicy(self.resource_id, {'ipsecpolicy': prop_diff})

    def handle_delete(self):
        try:
            self.client().delete_ipsecpolicy(self.resource_id)
        except Exception as ex:
            self.client_plugin().ignore_not_found(ex)
        else:
            return True