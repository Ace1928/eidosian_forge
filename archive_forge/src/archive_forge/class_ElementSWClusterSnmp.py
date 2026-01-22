from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_module import NetAppModule
class ElementSWClusterSnmp(object):
    """
    Element Software Configure Element SW Cluster SnmpNetwork
    """

    def __init__(self):
        self.argument_spec = netapp_utils.ontap_sf_host_argument_spec()
        self.argument_spec.update(dict(state=dict(type='str', choices=['present', 'absent'], default='present'), snmp_v3_enabled=dict(type='bool'), networks=dict(type='dict', options=dict(access=dict(type='str', choices=['ro', 'rw', 'rosys']), cidr=dict(type='int', default=None), community=dict(type='str', default=None), network=dict(type='str', default=None))), usm_users=dict(type='dict', options=dict(access=dict(type='str', choices=['rouser', 'rwuser', 'rosys']), name=dict(type='str', default=None), password=dict(type='str', default=None, no_log=True), passphrase=dict(type='str', default=None, no_log=True), secLevel=dict(type='str', choices=['auth', 'noauth', 'priv'])))))
        self.module = AnsibleModule(argument_spec=self.argument_spec, required_if=[('state', 'present', ['snmp_v3_enabled']), ('snmp_v3_enabled', True, ['usm_users']), ('snmp_v3_enabled', False, ['networks'])], supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        if self.parameters.get('state') == 'present':
            if self.parameters.get('usm_users') is not None:
                self.access_usm = self.parameters.get('usm_users')['access']
                self.name = self.parameters.get('usm_users')['name']
                self.password = self.parameters.get('usm_users')['password']
                self.passphrase = self.parameters.get('usm_users')['passphrase']
                self.secLevel = self.parameters.get('usm_users')['secLevel']
            if self.parameters.get('networks') is not None:
                self.access_network = self.parameters.get('networks')['access']
                self.cidr = self.parameters.get('networks')['cidr']
                self.community = self.parameters.get('networks')['community']
                self.network = self.parameters.get('networks')['network']
        if HAS_SF_SDK is False:
            self.module.fail_json(msg='Unable to import the SolidFire Python SDK')
        else:
            self.sfe = netapp_utils.create_sf_connection(module=self.module)

    def enable_snmp(self):
        """
        enable snmp feature
        """
        try:
            self.sfe.enable_snmp(snmp_v3_enabled=self.parameters.get('snmp_v3_enabled'))
        except Exception as exception_object:
            self.module.fail_json(msg='Error enabling snmp feature %s' % to_native(exception_object), exception=traceback.format_exc())

    def disable_snmp(self):
        """
        disable snmp feature
        """
        try:
            self.sfe.disable_snmp()
        except Exception as exception_object:
            self.module.fail_json(msg='Error disabling snmp feature %s' % to_native(exception_object), exception=traceback.format_exc())

    def configure_snmp(self, actual_networks, actual_usm_users):
        """
        Configure snmp
        """
        try:
            self.sfe.set_snmp_acl(networks=[actual_networks], usm_users=[actual_usm_users])
        except Exception as exception_object:
            self.module.fail_json(msg='Error Configuring snmp feature %s' % to_native(exception_object), exception=traceback.format_exc())

    def apply(self):
        """
        Cluster SNMP configuration
        """
        changed = False
        result_message = None
        update_required = False
        version_change = False
        is_snmp_enabled = self.sfe.get_snmp_state().enabled
        if is_snmp_enabled is True:
            if self.parameters.get('state') == 'absent':
                changed = True
            elif self.parameters.get('state') == 'present':
                is_snmp_v3_enabled = self.sfe.get_snmp_state().snmp_v3_enabled
                if is_snmp_v3_enabled != self.parameters.get('snmp_v3_enabled'):
                    version_change = True
                    changed = True
                if is_snmp_v3_enabled is True:
                    if len(self.sfe.get_snmp_info().usm_users) == 0:
                        update_required = True
                        changed = True
                    else:
                        for usm_user in self.sfe.get_snmp_info().usm_users:
                            if usm_user.access != self.access_usm or usm_user.name != self.name or usm_user.password != self.password or (usm_user.passphrase != self.passphrase) or (usm_user.sec_level != self.secLevel):
                                update_required = True
                                changed = True
                else:
                    for snmp_network in self.sfe.get_snmp_info().networks:
                        if snmp_network.access != self.access_network or snmp_network.cidr != self.cidr or snmp_network.community != self.community or (snmp_network.network != self.network):
                            update_required = True
                            changed = True
        elif self.parameters.get('state') == 'present':
            changed = True
        result_message = ''
        if changed:
            if self.module.check_mode is True:
                result_message = 'Check mode, skipping changes'
            elif self.parameters.get('state') == 'present':
                if self.parameters.get('snmp_v3_enabled') is True:
                    usm_users = {'access': self.access_usm, 'name': self.name, 'password': self.password, 'passphrase': self.passphrase, 'secLevel': self.secLevel}
                    networks = None
                else:
                    usm_users = None
                    networks = {'access': self.access_network, 'cidr': self.cidr, 'community': self.community, 'network': self.network}
                if is_snmp_enabled is False or version_change is True:
                    self.enable_snmp()
                    self.configure_snmp(networks, usm_users)
                    result_message = 'SNMP is enabled and configured'
                elif update_required is True:
                    self.configure_snmp(networks, usm_users)
                    result_message = 'SNMP is configured'
            elif is_snmp_enabled is True and self.parameters.get('state') == 'absent':
                self.disable_snmp()
                result_message = 'SNMP is disabled'
        self.module.exit_json(changed=changed, msg=result_message)