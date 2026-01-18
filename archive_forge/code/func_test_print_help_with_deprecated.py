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