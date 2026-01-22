import os.path
from unittest import mock
import fixtures
from oslo_config import cfg
from heat.common import environment_format
from heat.common import exception
from heat.engine import environment
from heat.engine import resources
from heat.engine.resources.aws.ec2 import instance
from heat.engine.resources.openstack.nova import server
from heat.engine import support
from heat.tests import common
from heat.tests import generic_resource
from heat.tests import utils
class GlobalEnvLoadingTest(common.HeatTestCase):

    def test_happy_path(self):
        with mock.patch('glob.glob') as m_ldir:
            m_ldir.return_value = ['/etc_etc/heat/environment.d/a.yaml']
            env_dir = '/etc_etc/heat/environment.d'
            env_content = '{"resource_registry": {}}'
            env = environment.Environment({}, user_env=False)
            with mock.patch('heat.engine.environment.open', mock.mock_open(read_data=env_content), create=True) as m_open:
                environment.read_global_environment(env, env_dir)
        m_ldir.assert_called_once_with(env_dir + '/*')
        m_open.assert_called_once_with('%s/a.yaml' % env_dir)

    def test_empty_env_dir(self):
        with mock.patch('glob.glob') as m_ldir:
            m_ldir.return_value = []
            env_dir = '/etc_etc/heat/environment.d'
            env = environment.Environment({}, user_env=False)
            environment.read_global_environment(env, env_dir)
        m_ldir.assert_called_once_with(env_dir + '/*')

    def test_continue_on_ioerror(self):
        """Assert we get all files processed.

        Assert we get all files processed even if there are processing
        exceptions.

        Test uses IOError as side effect of mock open.
        """
        with mock.patch('glob.glob') as m_ldir:
            m_ldir.return_value = ['/etc_etc/heat/environment.d/a.yaml', '/etc_etc/heat/environment.d/b.yaml']
            env_dir = '/etc_etc/heat/environment.d'
            env_content = '{}'
            env = environment.Environment({}, user_env=False)
            with mock.patch('heat.engine.environment.open', mock.mock_open(read_data=env_content), create=True) as m_open:
                m_open.side_effect = IOError
                environment.read_global_environment(env, env_dir)
        m_ldir.assert_called_once_with(env_dir + '/*')
        expected = [mock.call('%s/a.yaml' % env_dir), mock.call('%s/b.yaml' % env_dir)]
        self.assertEqual(expected, m_open.call_args_list)

    def test_continue_on_parse_error(self):
        """Assert we get all files processed.

        Assert we get all files processed even if there are processing
        exceptions.

        Test checks case when env content is incorrect.
        """
        with mock.patch('glob.glob') as m_ldir:
            m_ldir.return_value = ['/etc_etc/heat/environment.d/a.yaml', '/etc_etc/heat/environment.d/b.yaml']
            env_dir = '/etc_etc/heat/environment.d'
            env_content = '{@$%#$%'
            env = environment.Environment({}, user_env=False)
            with mock.patch('heat.engine.environment.open', mock.mock_open(read_data=env_content), create=True) as m_open:
                environment.read_global_environment(env, env_dir)
        m_ldir.assert_called_once_with(env_dir + '/*')
        expected = [mock.call('%s/a.yaml' % env_dir), mock.call('%s/b.yaml' % env_dir)]
        self.assertEqual(expected, m_open.call_args_list)

    def test_env_resources_override_plugins(self):
        g_env_content = '\n        resource_registry:\n          "OS::Nova::Server": "file:///not_really_here.yaml"\n        '
        envdir = self.useFixture(fixtures.TempDir())
        envfile = os.path.join(envdir.path, 'test.yaml')
        with open(envfile, 'w+') as ef:
            ef.write(g_env_content)
        cfg.CONF.set_override('environment_dir', envdir.path)
        g_env = environment.Environment({}, user_env=False)
        resources._load_global_environment(g_env)
        self.assertEqual('file:///not_really_here.yaml', g_env.get_resource_info('OS::Nova::Server').value)

    def test_env_one_resource_disable(self):
        g_env_content = '\n        resource_registry:\n            "OS::Nova::Server":\n        '
        envdir = self.useFixture(fixtures.TempDir())
        envfile = os.path.join(envdir.path, 'test.yaml')
        with open(envfile, 'w+') as ef:
            ef.write(g_env_content)
        cfg.CONF.set_override('environment_dir', envdir.path)
        g_env = environment.Environment({}, user_env=False)
        resources._load_global_environment(g_env)
        self.assertRaises(exception.EntityNotFound, g_env.get_resource_info, 'OS::Nova::Server')
        self.assertEqual(instance.Instance, g_env.get_resource_info('AWS::EC2::Instance').value)

    def test_env_multi_resources_disable(self):
        g_env_content = '\n        resource_registry:\n            "AWS::*":\n        '
        envdir = self.useFixture(fixtures.TempDir())
        envfile = os.path.join(envdir.path, 'test.yaml')
        with open(envfile, 'w+') as ef:
            ef.write(g_env_content)
        cfg.CONF.set_override('environment_dir', envdir.path)
        g_env = environment.Environment({}, user_env=False)
        resources._load_global_environment(g_env)
        self.assertRaises(exception.EntityNotFound, g_env.get_resource_info, 'AWS::EC2::Instance')
        self.assertEqual(server.Server, g_env.get_resource_info('OS::Nova::Server').value)

    def test_env_user_cant_disable_sys_resource(self):
        u_env_content = '\n        resource_registry:\n            "AWS::*":\n        '
        u_env = environment.Environment()
        u_env.load(environment_format.parse(u_env_content))
        self.assertEqual(instance.Instance, u_env.get_resource_info('AWS::EC2::Instance').value)

    def test_env_ignore_files_starting_dot(self):
        g_env_content = ''
        envdir = self.useFixture(fixtures.TempDir())
        with open(os.path.join(envdir.path, 'a.yaml'), 'w+') as ef:
            ef.write(g_env_content)
        with open(os.path.join(envdir.path, '.test.yaml'), 'w+') as ef:
            ef.write(g_env_content)
        with open(os.path.join(envdir.path, 'b.yaml'), 'w+') as ef:
            ef.write(g_env_content)
        cfg.CONF.set_override('environment_dir', envdir.path)
        g_env = environment.Environment({}, user_env=False)
        with mock.patch('heat.engine.environment.open', mock.mock_open(read_data=g_env_content), create=True) as m_open:
            resources._load_global_environment(g_env)
        expected = [mock.call('%s/a.yaml' % envdir.path), mock.call('%s/b.yaml' % envdir.path)]
        call_list = m_open.call_args_list
        expected.sort()
        call_list.sort()
        self.assertEqual(expected, call_list)