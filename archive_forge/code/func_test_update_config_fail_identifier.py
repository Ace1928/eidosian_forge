import os
import tempfile
import unittest
from webtest import TestApp
import pecan
from pecan.tests import PecanTestCase
def test_update_config_fail_identifier(self):
    """Fail when naming does not pass correctness"""
    from pecan import configuration
    bad_dict = {'bad name': 'value'}
    self.assertRaises(ValueError, configuration.Config, bad_dict)