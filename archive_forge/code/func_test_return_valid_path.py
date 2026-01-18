import os
import tempfile
import unittest
from webtest import TestApp
import pecan
from pecan.tests import PecanTestCase
def test_return_valid_path(self):
    __here__ = os.path.abspath(__file__)
    os.environ['PECAN_CONFIG'] = __here__
    assert self.get_conf_path_from_env() == __here__