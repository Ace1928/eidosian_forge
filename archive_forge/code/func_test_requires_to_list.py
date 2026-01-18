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
def test_requires_to_list(self):
    attrs = {'name': 'package', 'requires': iter(['other'])}
    dist = Distribution(attrs)
    self.assertIsInstance(dist.metadata.requires, list)