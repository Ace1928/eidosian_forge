import os
import sys
import unittest
from copy import copy
from unittest import mock
from distutils.errors import DistutilsPlatformError, DistutilsByteCompileError
from distutils.util import (get_platform, convert_path, change_root,
from distutils import util # used to patch _environ_checked
from distutils.sysconfig import get_config_vars
from distutils import sysconfig
from distutils.tests import support
import _osx_support
def test_strtobool(self):
    yes = ('y', 'Y', 'yes', 'True', 't', 'true', 'True', 'On', 'on', '1')
    no = ('n', 'no', 'f', 'false', 'off', '0', 'Off', 'No', 'N')
    for y in yes:
        self.assertTrue(strtobool(y))
    for n in no:
        self.assertFalse(strtobool(n))