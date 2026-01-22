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
class TestConfigArgparse(base.TestCase):

    def setUp(self):
        super(TestConfigArgparse, self).setUp()
        self.args = dict(auth_url='http://example.com/v2', username='user', password='password', project_name='project', region_name='region2', snack_type='cookie', os_auth_token='no-good-things')
        self.options = argparse.Namespace(**self.args)

    def test_get_one_bad_region_argparse(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        self.assertRaises(exceptions.ConfigException, c.get_one, cloud='_test-cloud_', argparse=self.options)

    def test_get_one_argparse(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        cc = c.get_one(cloud='_test_cloud_regions', argparse=self.options, validate=False)
        self.assertEqual(cc.region_name, 'region2')
        self.assertEqual(cc.snack_type, 'cookie')

    def test_get_one_precedence(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        kwargs = {'auth': {'username': 'testuser', 'password': 'authpass', 'project-id': 'testproject', 'auth_url': 'http://example.com/v2'}, 'region_name': 'kwarg_region', 'password': 'ansible_password', 'arbitrary': 'value'}
        args = dict(auth_url='http://example.com/v2', username='user', password='argpass', project_name='project', region_name='region2', snack_type='cookie')
        options = argparse.Namespace(**args)
        cc = c.get_one(argparse=options, **kwargs)
        self.assertEqual(cc.region_name, 'region2')
        self.assertEqual(cc.auth['password'], 'authpass')
        self.assertEqual(cc.snack_type, 'cookie')

    def test_get_one_cloud_precedence_osc(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        kwargs = {'auth': {'username': 'testuser', 'password': 'authpass', 'project-id': 'testproject', 'auth_url': 'http://example.com/v2'}, 'region_name': 'kwarg_region', 'password': 'ansible_password', 'arbitrary': 'value'}
        args = dict(auth_url='http://example.com/v2', username='user', password='argpass', project_name='project', region_name='region2', snack_type='cookie')
        options = argparse.Namespace(**args)
        cc = c.get_one_cloud_osc(argparse=options, **kwargs)
        self.assertEqual(cc.region_name, 'region2')
        self.assertEqual(cc.auth['password'], 'argpass')
        self.assertEqual(cc.snack_type, 'cookie')

    def test_get_one_precedence_no_argparse(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        kwargs = {'auth': {'username': 'testuser', 'password': 'authpass', 'project-id': 'testproject', 'auth_url': 'http://example.com/v2'}, 'region_name': 'kwarg_region', 'password': 'ansible_password', 'arbitrary': 'value'}
        cc = c.get_one(**kwargs)
        self.assertEqual(cc.region_name, 'kwarg_region')
        self.assertEqual(cc.auth['password'], 'authpass')
        self.assertIsNone(cc.password)

    def test_get_one_just_argparse(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        cc = c.get_one(argparse=self.options, validate=False)
        self.assertIsNone(cc.cloud)
        self.assertEqual(cc.region_name, 'region2')
        self.assertEqual(cc.snack_type, 'cookie')

    def test_get_one_just_kwargs(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        cc = c.get_one(validate=False, **self.args)
        self.assertIsNone(cc.cloud)
        self.assertEqual(cc.region_name, 'region2')
        self.assertEqual(cc.snack_type, 'cookie')

    def test_get_one_dash_kwargs(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        args = {'auth-url': 'http://example.com/v2', 'username': 'user', 'password': 'password', 'project_name': 'project', 'region_name': 'other-test-region', 'snack_type': 'cookie'}
        cc = c.get_one(**args)
        self.assertIsNone(cc.cloud)
        self.assertEqual(cc.region_name, 'other-test-region')
        self.assertEqual(cc.snack_type, 'cookie')

    def test_get_one_no_argparse(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        cc = c.get_one(cloud='_test-cloud_', argparse=None)
        self._assert_cloud_details(cc)
        self.assertEqual(cc.region_name, 'test-region')
        self.assertIsNone(cc.snack_type)

    def test_get_one_no_argparse_regions(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        cc = c.get_one(cloud='_test_cloud_regions', argparse=None)
        self._assert_cloud_details(cc)
        self.assertEqual(cc.region_name, 'region1')
        self.assertIsNone(cc.snack_type)

    def test_get_one_bad_region(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        self.assertRaises(exceptions.ConfigException, c.get_one, cloud='_test_cloud_regions', region_name='bad')

    def test_get_one_bad_region_no_regions(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        self.assertRaises(exceptions.ConfigException, c.get_one, cloud='_test-cloud_', region_name='bad_region')

    def test_get_one_no_argparse_region2(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        cc = c.get_one(cloud='_test_cloud_regions', region_name='region2', argparse=None)
        self._assert_cloud_details(cc)
        self.assertEqual(cc.region_name, 'region2')
        self.assertIsNone(cc.snack_type)

    def test_get_one_network(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        cc = c.get_one(cloud='_test_cloud_regions', region_name='region1', argparse=None)
        self._assert_cloud_details(cc)
        self.assertEqual(cc.region_name, 'region1')
        self.assertEqual('region1-network', cc.config['external_network'])

    def test_get_one_per_region_network(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        cc = c.get_one(cloud='_test_cloud_regions', region_name='region2', argparse=None)
        self._assert_cloud_details(cc)
        self.assertEqual(cc.region_name, 'region2')
        self.assertEqual('my-network', cc.config['external_network'])

    def test_get_one_no_yaml_no_cloud(self):
        c = config.OpenStackConfig(load_yaml_config=False)
        self.assertRaises(exceptions.ConfigException, c.get_one, cloud='_test_cloud_regions', region_name='region2', argparse=None)

    def test_get_one_no_yaml(self):
        c = config.OpenStackConfig(load_yaml_config=False)
        cc = c.get_one(region_name='region2', argparse=None, **base.USER_CONF['clouds']['_test_cloud_regions'])
        self.assertIsInstance(cc, cloud_region.CloudRegion)
        self.assertTrue(hasattr(cc, 'auth'))
        self.assertIsInstance(cc.auth, dict)
        self.assertIsNone(cc.cloud)
        self.assertIn('username', cc.auth)
        self.assertEqual('testuser', cc.auth['username'])
        self.assertEqual('testpass', cc.auth['password'])
        self.assertFalse(cc.config['image_api_use_tasks'])
        self.assertTrue('project_name' in cc.auth or 'project_id' in cc.auth)
        if 'project_name' in cc.auth:
            self.assertEqual('testproject', cc.auth['project_name'])
        elif 'project_id' in cc.auth:
            self.assertEqual('testproject', cc.auth['project_id'])
        self.assertEqual(cc.region_name, 'region2')

    def test_fix_env_args(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        env_args = {'os-compute-api-version': 1}
        fixed_args = c._fix_args(env_args)
        self.assertDictEqual({'compute_api_version': 1}, fixed_args)

    def test_extra_config(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        defaults = {'use_hostnames': False, 'other-value': 'something'}
        ansible_options = c.get_extra_config('ansible', defaults)
        self.assertDictEqual({'expand_hostvars': False, 'use_hostnames': True, 'other_value': 'something'}, ansible_options)

    def test_get_client_config(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        cc = c.get_one(cloud='_test_cloud_regions')
        defaults = {'use_hostnames': False, 'other-value': 'something', 'force_ipv4': False}
        ansible_options = cc.get_client_config('ansible', defaults)
        self.assertDictEqual({'expand_hostvars': False, 'use_hostnames': True, 'other_value': 'something', 'force_ipv4': True}, ansible_options)

    def test_register_argparse_cloud(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        parser = argparse.ArgumentParser()
        c.register_argparse_arguments(parser, [])
        opts, _remain = parser.parse_known_args(['--os-cloud', 'foo'])
        self.assertEqual(opts.os_cloud, 'foo')

    def test_env_argparse_precedence(self):
        self.useFixture(fixtures.EnvironmentVariable('OS_TENANT_NAME', 'tenants-are-bad'))
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        cc = c.get_one(cloud='envvars', argparse=self.options, validate=False)
        self.assertEqual(cc.auth['project_name'], 'project')

    def test_argparse_default_no_token(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        parser = argparse.ArgumentParser()
        c.register_argparse_arguments(parser, [])
        parser.add_argument('--os-auth-token')
        opts, _remain = parser.parse_known_args()
        cc = c.get_one(cloud='_test_cloud_regions', argparse=opts)
        self.assertEqual(cc.config['auth_type'], 'password')
        self.assertNotIn('token', cc.config['auth'])

    def test_argparse_token(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        parser = argparse.ArgumentParser()
        c.register_argparse_arguments(parser, [])
        parser.add_argument('--os-auth-token')
        opts, _remain = parser.parse_known_args(['--os-auth-token', 'very-bad-things', '--os-auth-type', 'token'])
        cc = c.get_one(argparse=opts, validate=False)
        self.assertEqual(cc.config['auth_type'], 'token')
        self.assertEqual(cc.config['auth']['token'], 'very-bad-things')

    def test_argparse_underscores(self):
        c = config.OpenStackConfig(config_files=[self.no_yaml], vendor_files=[self.no_yaml], secure_files=[self.no_yaml])
        parser = argparse.ArgumentParser()
        parser.add_argument('--os_username')
        argv = ['--os_username', 'user', '--os_password', 'pass', '--os-auth-url', 'auth-url', '--os-project-name', 'project']
        c.register_argparse_arguments(parser, argv=argv)
        opts, _remain = parser.parse_known_args(argv)
        cc = c.get_one(argparse=opts)
        self.assertEqual(cc.config['auth']['username'], 'user')
        self.assertEqual(cc.config['auth']['password'], 'pass')
        self.assertEqual(cc.config['auth']['auth_url'], 'auth-url')

    def test_argparse_action_append_no_underscore(self):
        c = config.OpenStackConfig(config_files=[self.no_yaml], vendor_files=[self.no_yaml], secure_files=[self.no_yaml])
        parser = argparse.ArgumentParser()
        parser.add_argument('--foo', action='append')
        argv = ['--foo', '1', '--foo', '2']
        c.register_argparse_arguments(parser, argv=argv)
        opts, _remain = parser.parse_known_args(argv)
        self.assertEqual(opts.foo, ['1', '2'])

    def test_argparse_underscores_duplicate(self):
        c = config.OpenStackConfig(config_files=[self.no_yaml], vendor_files=[self.no_yaml], secure_files=[self.no_yaml])
        parser = argparse.ArgumentParser()
        parser.add_argument('--os_username')
        argv = ['--os_username', 'user', '--os_password', 'pass', '--os-username', 'user1', '--os-password', 'pass1', '--os-auth-url', 'auth-url', '--os-project-name', 'project']
        self.assertRaises(exceptions.ConfigException, c.register_argparse_arguments, parser=parser, argv=argv)

    def test_register_argparse_bad_plugin(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        parser = argparse.ArgumentParser()
        self.assertRaises(exceptions.ConfigException, c.register_argparse_arguments, parser, ['--os-auth-type', 'foo'])

    def test_register_argparse_not_password(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        parser = argparse.ArgumentParser()
        args = ['--os-auth-type', 'v3token', '--os-token', 'some-secret']
        c.register_argparse_arguments(parser, args)
        opts, _remain = parser.parse_known_args(args)
        self.assertEqual(opts.os_token, 'some-secret')

    def test_register_argparse_password(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        parser = argparse.ArgumentParser()
        args = ['--os-password', 'some-secret']
        c.register_argparse_arguments(parser, args)
        opts, _remain = parser.parse_known_args(args)
        self.assertEqual(opts.os_password, 'some-secret')
        with testtools.ExpectedException(AttributeError):
            opts.os_token

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

    def test_register_argparse_network_service_type(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        parser = argparse.ArgumentParser()
        args = ['--os-endpoint-type', 'admin', '--network-api-version', '4']
        c.register_argparse_arguments(parser, args, ['network'])
        opts, _remain = parser.parse_known_args(args)
        self.assertEqual(opts.os_service_type, 'network')
        self.assertEqual(opts.os_endpoint_type, 'admin')
        self.assertIsNone(opts.os_network_service_type)
        self.assertIsNone(opts.os_network_api_version)
        self.assertEqual(opts.network_api_version, '4')
        cloud = c.get_one(argparse=opts, validate=False)
        self.assertEqual(cloud.config['service_type'], 'network')
        self.assertEqual(cloud.config['interface'], 'admin')
        self.assertEqual(cloud.config['network_api_version'], '4')
        self.assertNotIn('http_timeout', cloud.config)

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