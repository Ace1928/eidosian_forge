import os
import tempfile
import unittest
from webtest import TestApp
import pecan
from pecan.tests import PecanTestCase
def test_paint_from_file(self):
    from pecan import configuration
    configuration.set_config(os.path.join(__here__, 'config_fixtures/foobar.py'))
    assert dict(configuration._runtime_conf) != {'foo': 'bar'}
    assert configuration._runtime_conf.foo == 'bar'