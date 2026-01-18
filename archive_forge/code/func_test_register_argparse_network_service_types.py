import argparse
import copy
import os
from unittest import mock
import fixtures
import testtools
import yaml
from openstack import config
from openstack.config import cloud_region
from openstack.config import defaults
from openstack import exceptions
from openstack.tests.unit.config import base
def test_register_argparse_network_service_types(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    parser = argparse.ArgumentParser()
    args = ['--os-compute-service-name', 'cloudServers', '--os-network-service-type', 'badtype', '--os-endpoint-type', 'admin', '--network-api-version', '4']
    c.register_argparse_arguments(parser, args, ['compute', 'network', 'volume'])
    opts, _remain = parser.parse_known_args(args)
    self.assertEqual(opts.os_network_service_type, 'badtype')
    self.assertIsNone(opts.os_compute_service_type)
    self.assertIsNone(opts.os_volume_service_type)
    self.assertEqual(opts.os_service_type, 'compute')
    self.assertEqual(opts.os_compute_service_name, 'cloudServers')
    self.assertEqual(opts.os_endpoint_type, 'admin')
    self.assertIsNone(opts.os_network_api_version)
    self.assertEqual(opts.network_api_version, '4')
    cloud = c.get_one(argparse=opts, validate=False)
    self.assertEqual(cloud.config['service_type'], 'compute')
    self.assertEqual(cloud.config['network_service_type'], 'badtype')
    self.assertEqual(cloud.config['interface'], 'admin')
    self.assertEqual(cloud.config['network_api_version'], '4')
    self.assertNotIn('volume_service_type', cloud.config)
    self.assertNotIn('http_timeout', cloud.config)