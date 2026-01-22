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
class RegisterOptNameTestCase(BaseTestCase):

    def test_register_opt_with_disallow_name(self):
        for name in cfg.ConfigOpts.disallow_names:
            opt = cfg.StrOpt(name)
            self.assertRaises(ValueError, self.conf.register_opt, opt)