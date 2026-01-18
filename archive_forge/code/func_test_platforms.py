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
def test_platforms(self):
    attrs = {'name': 'Monty', 'version': '1.0', 'platforms': ['GNU/Linux', 'Some Evil Platform']}
    dist = Distribution(attrs)
    self.assertEqual(dist.get_platforms(), ['GNU/Linux', 'Some Evil Platform'])