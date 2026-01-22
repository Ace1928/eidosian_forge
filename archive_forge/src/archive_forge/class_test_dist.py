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
class test_dist(Command):
    """Sample distutils extension command."""
    user_options = [('sample-option=', 'S', 'help text')]

    def initialize_options(self):
        self.sample_option = None