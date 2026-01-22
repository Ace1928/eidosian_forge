import argparse
import errno
import functools
import io
import logging
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock
import fixtures
from oslotest import base
import testscenarios
from oslo_config import cfg
from oslo_config import types
class HelpTestCase(BaseTestCase):

    def test_print_help(self):
        f = io.StringIO()
        self.conf([])
        self.conf.print_help(file=f)
        self.assertIn('usage: test [-h] [--config-dir DIR] [--config-file PATH] [--version]', f.getvalue())
        self.assertRegex(f.getvalue(), 'option(s|al arguments):')
        self.assertIn('-h, --help', f.getvalue())

    def test_print_strOpt_with_choices_help(self):
        f = io.StringIO()
        cli_opts = [cfg.StrOpt('aa', short='a', default='xx', choices=['xx', 'yy', 'zz'], help='StrOpt with choices.'), cfg.StrOpt('bb', short='b', default='yy', choices=[None, 'yy', 'zz'], help='StrOpt with choices.'), cfg.StrOpt('cc', short='c', default='zz', choices=['', 'yy', 'zz'], help='StrOpt with choices.')]
        self.conf.register_cli_opts(cli_opts)
        self.conf([])
        self.conf.print_help(file=f)
        self.assertIn('usage: test [-h] [--aa AA] [--bb BB] [--cc CC] [--config-dir DIR]', f.getvalue())
        self.assertRegex(f.getvalue(), 'option(s|al arguments):')
        self.assertIn('-h, --help', f.getvalue())
        self.assertIn('StrOpt with choices. Allowed values: xx, yy, zz', f.getvalue())
        self.assertIn('StrOpt with choices. Allowed values: <None>, yy, zz', f.getvalue())
        self.assertIn("StrOpt with choices. Allowed values: '', yy, zz", f.getvalue())

    def test_print_sorted_help(self):
        f = io.StringIO()
        self.conf.register_cli_opt(cfg.StrOpt('abc'))
        self.conf.register_cli_opt(cfg.StrOpt('zba'))
        self.conf.register_cli_opt(cfg.StrOpt('ghi'))
        self.conf.register_cli_opt(cfg.StrOpt('deb'))
        self.conf([])
        self.conf.print_help(file=f)
        zba = f.getvalue().find('--zba')
        abc = f.getvalue().find('--abc')
        ghi = f.getvalue().find('--ghi')
        deb = f.getvalue().find('--deb')
        list = [abc, deb, ghi, zba]
        self.assertEqual(sorted(list), list)

    def test_print_sorted_help_with_positionals(self):
        f = io.StringIO()
        self.conf.register_cli_opt(cfg.StrOpt('pst', positional=True, required=False))
        self.conf.register_cli_opt(cfg.StrOpt('abc'))
        self.conf.register_cli_opt(cfg.StrOpt('zba'))
        self.conf.register_cli_opt(cfg.StrOpt('ghi'))
        self.conf([])
        self.conf.print_help(file=f)
        zba = f.getvalue().find('--zba')
        abc = f.getvalue().find('--abc')
        ghi = f.getvalue().find('--ghi')
        list = [abc, ghi, zba]
        self.assertEqual(sorted(list), list)

    def test_print_help_with_deprecated(self):
        f = io.StringIO()
        abc = cfg.StrOpt('a-bc', deprecated_opts=[cfg.DeprecatedOpt('d-ef')])
        uvw = cfg.StrOpt('u-vw', deprecated_name='x-yz')
        self.conf.register_cli_opt(abc)
        self.conf.register_cli_opt(uvw)
        self.conf([])
        self.conf.print_help(file=f)
        self.assertIn('--a-bc A_BC, --d-ef A_BC, --d_ef A_BC', f.getvalue())
        self.assertIn('--u-vw U_VW, --x-yz U_VW, --x_yz U_VW', f.getvalue())