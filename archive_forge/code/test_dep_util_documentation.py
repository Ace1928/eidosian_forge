import unittest
import os
from distutils.dep_util import newer, newer_pairwise, newer_group
from distutils.errors import DistutilsFileError
from distutils.tests import support
Tests for distutils.dep_util.