import os
import tempfile
import unittest
from webtest import TestApp
import pecan
from pecan.tests import PecanTestCase
def _remove_config_key(self):
    os.environ.pop('PECAN_CONFIG', None)