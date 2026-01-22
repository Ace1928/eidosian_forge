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
def _do_dgroup_and_dname_test_ignore(self, opt_class, value, result):
    self._do_deprecated_test(opt_class, value, result, 'oof', section='old', dgroup='old', dname='oof')