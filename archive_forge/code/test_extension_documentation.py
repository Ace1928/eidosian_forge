import unittest
import os
import warnings
from test.support.warnings_helper import check_warnings
from distutils.extension import read_setup_file, Extension
Tests for distutils.extension.