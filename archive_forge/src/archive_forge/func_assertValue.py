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
def assertValue(self, key, expect, multi=False, normalized=False):
    actual, _ = self.ns._get_value([key], multi=multi, normalized=normalized)
    self.assertEqual(actual, expect)