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