import unittest
import os
import errno
from unittest.mock import patch
from distutils.file_util import move_file, copy_file
from distutils import log
from distutils.tests import support
from distutils.errors import DistutilsFileError
from test.support.os_helper import unlink
Tests for distutils.file_util.