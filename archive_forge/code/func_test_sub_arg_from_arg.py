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
def test_sub_arg_from_arg(self):
    self._prep_test_str_int_sub()
    self.conf(['--foo', '123', '--bar', '$foo'])
    self._assert_int_sub()