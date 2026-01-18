import os
import tempfile
import unittest
from webtest import TestApp
import pecan
from pecan.tests import PecanTestCase
def test_paint_from_dict(self):
    from pecan import configuration
    configuration.set_config({'foo': 'bar'})
    assert dict(configuration._runtime_conf) != {'foo': 'bar'}
    self.assertEqual(configuration._runtime_conf.foo, 'bar')