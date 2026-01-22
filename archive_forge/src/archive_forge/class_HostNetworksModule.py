from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
class HostNetworksModule(BaseModule):

    def __compare_options(self, new_options, old_options):
        return sorted((get_dict_of_struct(opt) for opt in new_options), key=lambda x: x['name']) != sorted((get_dict_of_struct(opt) for opt in old_options), key=lambda x: x['name'])

    def build_entity(self):
        return otypes.Host()

    def update_custom_properties(self, attachments_service, attachment, network):
        if network.get('custom_properties'):
            current = []
            if attachment.properties:
                current = [(cp.name, str(cp.value)) for cp in attachment.properties]
            passed = [(cp.get('name'), str(cp.get('value'))) for cp in network.get('custom_properties') if cp]
            if sorted(current) != sorted(passed):
                attachment.properties = [otypes.Property(name=prop.get('name'), value=prop.get('value')) for prop in network.get('custom_properties')]
                if not self._module.check_mode:
                    attachments_service.service(attachment.id).update(attachment)
                self.changed = True

    def update_address(self, attachments_service, attachment, network):
        for ip in attachment.ip_address_assignments:
            if str(ip.ip.version) == network.get('version', 'v4'):
                changed = False
                if not equal(network.get('boot_protocol'), str(ip.assignment_method)):
                    ip.assignment_method = otypes.BootProtocol(network.get('boot_protocol'))
                    changed = True
                if not equal(network.get('address'), ip.ip.address):
                    ip.ip.address = network.get('address')
                    changed = True
                if not equal(network.get('gateway'), ip.ip.gateway):
                    ip.ip.gateway = network.get('gateway')
                    changed = True
                if not equal(network.get('netmask'), ip.ip.netmask):
                    ip.ip.netmask = network.get('netmask')
                    changed = True
                if changed:
                    if not self._module.check_mode:
                        attachments_service.service(attachment.id).update(attachment)
                    self.changed = True
                    break

    def has_update(self, nic_service):
        update = False
        bond = self._module.params['bond']
        networks = self._module.params['networks']
        labels = self._module.params['labels']
        nic = get_entity(nic_service)
        if nic is None:
            return update
        if bond:
            update = self.__compare_options(get_bond_options(bond.get('mode'), bond.get('options')), getattr(nic.bonding, 'options', []))
            update = update or not equal(sorted(bond.get('interfaces')) if bond.get('interfaces') else None, sorted((get_link_name(self._connection, s) for s in nic.bonding.slaves)))
        if labels:
            net_labels = nic_service.network_labels_service().list()
            if sorted(labels) != sorted([lbl.id for lbl in net_labels]):
                return True
        if not networks:
            return update
        attachments_service = nic_service.network_attachments_service()
        network_names = [network.get('name') for network in networks]
        attachments = {}
        for attachment in attachments_service.list():
            name = get_link_name(self._connection, attachment.network)
            if name in network_names:
                attachments[name] = attachment
        for network in networks:
            attachment = attachments.get(network.get('name'))
            if attachment is None:
                return True
            self.update_custom_properties(attachments_service, attachment, network)
            self.update_address(attachments_service, attachment, network)
        return update

    def _action_save_configuration(self, entity):
        if not self._module.check_mode:
            self._service.service(entity.id).commit_net_config()
        self.changed = True