from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
class EntityNicsModule(BaseModule):

    def __init__(self, *args, **kwargs):
        super(EntityNicsModule, self).__init__(*args, **kwargs)
        self.vnic_id = None

    @property
    def vnic_id(self):
        return self._vnic_id

    @vnic_id.setter
    def vnic_id(self, vnic_id):
        self._vnic_id = vnic_id

    def post_create(self, entity):
        self._set_network_filter_parameters(entity.id)

    def post_update(self, entity):
        self._set_network_filter_parameters(entity.id)

    def _set_network_filter_parameters(self, entity_id):
        if self._module.params['network_filter_parameters'] is not None:
            nfps_service = self._service.service(entity_id).network_filter_parameters_service()
            nfp_list = nfps_service.list()
            for nfp in nfp_list:
                nfps_service.service(nfp.id).remove()
            for nfp in self._network_filter_parameters():
                nfps_service.add(nfp)

    def build_entity(self):
        return otypes.Nic(id=self._module.params.get('id'), name=self._module.params.get('name'), interface=otypes.NicInterface(self._module.params.get('interface')) if self._module.params.get('interface') else None, vnic_profile=otypes.VnicProfile(id=self.vnic_id) if self.vnic_id else None, mac=otypes.Mac(address=self._module.params.get('mac_address')) if self._module.params.get('mac_address') else None, linked=self.param('linked') if self.param('linked') is not None else None)

    def update_check(self, entity):
        if self._module.params.get('vm'):
            return equal(self._module.params.get('interface'), str(entity.interface)) and equal(self._module.params.get('linked'), entity.linked) and equal(self._module.params.get('name'), str(entity.name)) and equal(self._module.params.get('profile'), get_link_name(self._connection, entity.vnic_profile)) and equal(self._module.params.get('mac_address'), entity.mac.address) and equal(self._network_filter_parameters(), self._connection.follow_link(entity.network_filter_parameters))
        elif self._module.params.get('template'):
            return equal(self._module.params.get('interface'), str(entity.interface)) and equal(self._module.params.get('linked'), entity.linked) and equal(self._module.params.get('name'), str(entity.name)) and equal(self._module.params.get('profile'), get_link_name(self._connection, entity.vnic_profile))

    def _network_filter_parameters(self):
        if self._module.params['network_filter_parameters'] is None:
            return []
        networkFilterParameters = list()
        for networkFilterParameter in self._module.params['network_filter_parameters']:
            networkFilterParameters.append(otypes.NetworkFilterParameter(name=networkFilterParameter.get('name'), value=networkFilterParameter.get('value')))
        return networkFilterParameters