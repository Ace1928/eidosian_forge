from openstack.tests.functional import base
def _set_network_external(self, networks):
    for network in networks:
        if network.name == 'public':
            self.operator_cloud.network.update_network(network, is_default=True)