import os
import io
import sys
import unittest
import warnings
import textwrap
from unittest import mock
from distutils.dist import Distribution, fix_help_options
from distutils.cmd import Command
from test.support import (
from test.support.os_helper import TESTFN
from distutils.tests import support
from distutils import log
class DistributionTestCase(support.LoggingSilencer, support.TempdirManager, support.EnvironGuard, unittest.TestCase):

    def setUp(self):
        super(DistributionTestCase, self).setUp()
        self.argv = (sys.argv, sys.argv[:])
        del sys.argv[1:]

    def tearDown(self):
        sys.argv = self.argv[0]
        sys.argv[:] = self.argv[1]
        super(DistributionTestCase, self).tearDown()

    def create_distribution(self, configfiles=()):
        d = TestDistribution()
        d._config_files = configfiles
        d.parse_config_files()
        d.parse_command_line()
        return d

    def test_command_packages_unspecified(self):
        sys.argv.append('build')
        d = self.create_distribution()
        self.assertEqual(d.get_command_packages(), ['distutils.command'])

    def test_command_packages_cmdline(self):
        from distutils.tests.test_dist import test_dist
        sys.argv.extend(['--command-packages', 'foo.bar,distutils.tests', 'test_dist', '-Ssometext'])
        d = self.create_distribution()
        self.assertEqual(d.get_command_packages(), ['distutils.command', 'foo.bar', 'distutils.tests'])
        cmd = d.get_command_obj('test_dist')
        self.assertIsInstance(cmd, test_dist)
        self.assertEqual(cmd.sample_option, 'sometext')

    def test_venv_install_options(self):
        sys.argv.append('install')
        self.addCleanup(os.unlink, TESTFN)
        fakepath = '/somedir'
        with open(TESTFN, 'w') as f:
            print('[install]\ninstall-base = {0}\ninstall-platbase = {0}\ninstall-lib = {0}\ninstall-platlib = {0}\ninstall-purelib = {0}\ninstall-headers = {0}\ninstall-scripts = {0}\ninstall-data = {0}\nprefix = {0}\nexec-prefix = {0}\nhome = {0}\nuser = {0}\nroot = {0}'.format(fakepath), file=f)
        with mock.patch.multiple(sys, prefix='/a', base_prefix='/a') as values:
            d = self.create_distribution([TESTFN])
        option_tuple = (TESTFN, fakepath)
        result_dict = {'install_base': option_tuple, 'install_platbase': option_tuple, 'install_lib': option_tuple, 'install_platlib': option_tuple, 'install_purelib': option_tuple, 'install_headers': option_tuple, 'install_scripts': option_tuple, 'install_data': option_tuple, 'prefix': option_tuple, 'exec_prefix': option_tuple, 'home': option_tuple, 'user': option_tuple, 'root': option_tuple}
        self.assertEqual(sorted(d.command_options.get('install').keys()), sorted(result_dict.keys()))
        for key, value in d.command_options.get('install').items():
            self.assertEqual(value, result_dict[key])
        with mock.patch.multiple(sys, prefix='/a', base_prefix='/b') as values:
            d = self.create_distribution([TESTFN])
        for key in result_dict.keys():
            self.assertNotIn(key, d.command_options.get('install', {}))

    def test_command_packages_configfile(self):
        sys.argv.append('build')
        self.addCleanup(os.unlink, TESTFN)
        f = open(TESTFN, 'w')
        try:
            print('[global]', file=f)
            print('command_packages = foo.bar, splat', file=f)
        finally:
            f.close()
        d = self.create_distribution([TESTFN])
        self.assertEqual(d.get_command_packages(), ['distutils.command', 'foo.bar', 'splat'])
        sys.argv[1:] = ['--command-packages', 'spork', 'build']
        d = self.create_distribution([TESTFN])
        self.assertEqual(d.get_command_packages(), ['distutils.command', 'spork'])
        sys.argv[1:] = ['--command-packages', '', 'build']
        d = self.create_distribution([TESTFN])
        self.assertEqual(d.get_command_packages(), ['distutils.command'])

    def test_empty_options(self):
        warns = []

        def _warn(msg):
            warns.append(msg)
        self.addCleanup(setattr, warnings, 'warn', warnings.warn)
        warnings.warn = _warn
        dist = Distribution(attrs={'author': 'xxx', 'name': 'xxx', 'version': 'xxx', 'url': 'xxxx', 'options': {}})
        self.assertEqual(len(warns), 0)
        self.assertNotIn('options', dir(dist))

    def test_finalize_options(self):
        attrs = {'keywords': 'one,two', 'platforms': 'one,two'}
        dist = Distribution(attrs=attrs)
        dist.finalize_options()
        self.assertEqual(dist.metadata.platforms, ['one', 'two'])
        self.assertEqual(dist.metadata.keywords, ['one', 'two'])
        attrs = {'keywords': 'foo bar', 'platforms': 'foo bar'}
        dist = Distribution(attrs=attrs)
        dist.finalize_options()
        self.assertEqual(dist.metadata.platforms, ['foo bar'])
        self.assertEqual(dist.metadata.keywords, ['foo bar'])

    def test_get_command_packages(self):
        dist = Distribution()
        self.assertEqual(dist.command_packages, None)
        cmds = dist.get_command_packages()
        self.assertEqual(cmds, ['distutils.command'])
        self.assertEqual(dist.command_packages, ['distutils.command'])
        dist.command_packages = 'one,two'
        cmds = dist.get_command_packages()
        self.assertEqual(cmds, ['distutils.command', 'one', 'two'])

    def test_announce(self):
        dist = Distribution()
        args = ('ok',)
        kwargs = {'level': 'ok2'}
        self.assertRaises(ValueError, dist.announce, args, kwargs)

    def test_find_config_files_disable(self):
        temp_home = self.mkdtemp()
        if os.name == 'posix':
            user_filename = os.path.join(temp_home, '.pydistutils.cfg')
        else:
            user_filename = os.path.join(temp_home, 'pydistutils.cfg')
        with open(user_filename, 'w') as f:
            f.write('[distutils]\n')

        def _expander(path):
            return temp_home
        old_expander = os.path.expanduser
        os.path.expanduser = _expander
        try:
            d = Distribution()
            all_files = d.find_config_files()
            d = Distribution(attrs={'script_args': ['--no-user-cfg']})
            files = d.find_config_files()
        finally:
            os.path.expanduser = old_expander
        self.assertEqual(len(all_files) - 1, len(files))