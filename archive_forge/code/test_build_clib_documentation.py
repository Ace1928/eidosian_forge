import unittest
import os
import sys
import sysconfig
from test.support import (
from distutils.command.build_clib import build_clib
from distutils.errors import DistutilsSetupError
from distutils.tests import support
Tests for distutils.command.build_clib.