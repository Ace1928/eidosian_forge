from unittest import mock
from magnumclient.common.apiclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.v1.cluster_templates import ClusterTemplate
def _get_expected_args(self, image_id, external_network_id, coe, master_flavor_id=None, name=None, keypair_id=None, fixed_network=None, fixed_subnet=None, network_driver=None, volume_driver=None, dns_nameserver='8.8.8.8', flavor_id='m1.medium', docker_storage_driver='devicemapper', docker_volume_size=None, http_proxy=None, https_proxy=None, no_proxy=None, labels={}, tls_disabled=False, public=False, master_lb_enabled=False, server_type='vm', registry_enabled=False, insecure_registry=None, hidden=False):
    expected_args = {}
    expected_args['image_id'] = image_id
    expected_args['external_network_id'] = external_network_id
    expected_args['coe'] = coe
    expected_args['master_flavor_id'] = master_flavor_id
    expected_args['name'] = name
    expected_args['keypair_id'] = keypair_id
    expected_args['fixed_network'] = fixed_network
    expected_args['fixed_subnet'] = fixed_subnet
    expected_args['network_driver'] = network_driver
    expected_args['volume_driver'] = volume_driver
    expected_args['dns_nameserver'] = dns_nameserver
    expected_args['flavor_id'] = flavor_id
    expected_args['docker_volume_size'] = docker_volume_size
    expected_args['docker_storage_driver'] = docker_storage_driver
    expected_args['http_proxy'] = http_proxy
    expected_args['https_proxy'] = https_proxy
    expected_args['no_proxy'] = no_proxy
    expected_args['labels'] = labels
    expected_args['tls_disabled'] = tls_disabled
    expected_args['public'] = public
    expected_args['master_lb_enabled'] = master_lb_enabled
    expected_args['server_type'] = server_type
    expected_args['registry_enabled'] = registry_enabled
    expected_args['insecure_registry'] = insecure_registry
    expected_args['hidden'] = hidden
    return expected_args