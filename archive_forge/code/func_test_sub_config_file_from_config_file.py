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
def test_sub_config_file_from_config_file(self):
    self._prep_test_str_int_sub()
    paths = self.create_tempfiles([('test', '[DEFAULT]\nbar = $foo\nfoo = 123\n')])
    self.conf(['--config-file', paths[0]])
    self._assert_int_sub()