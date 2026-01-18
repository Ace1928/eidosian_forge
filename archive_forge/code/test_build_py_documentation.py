import os
import sys
import unittest
from distutils.command.build_py import build_py
from distutils.core import Distribution
from distutils.errors import DistutilsFileError
from distutils.tests import support
from test.support import requires_subprocess

        A directory in package_data should not be added to the filelist.
        