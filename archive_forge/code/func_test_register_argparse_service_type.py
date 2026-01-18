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
def test_register_argparse_service_type(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    parser = argparse.ArgumentParser()
    args = ['--os-service-type', 'network', '--os-endpoint-type', 'admin', '--http-timeout', '20']
    c.register_argparse_arguments(parser, args)
    opts, _remain = parser.parse_known_args(args)
    self.assertEqual(opts.os_service_type, 'network')
    self.assertEqual(opts.os_endpoint_type, 'admin')
    self.assertEqual(opts.http_timeout, '20')
    with testtools.ExpectedException(AttributeError):
        opts.os_network_service_type
    cloud = c.get_one(argparse=opts, validate=False)
    self.assertEqual(cloud.config['service_type'], 'network')
    self.assertEqual(cloud.config['interface'], 'admin')
    self.assertEqual(cloud.config['api_timeout'], '20')
    self.assertNotIn('http_timeout', cloud.config)