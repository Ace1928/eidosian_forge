import os
import tempfile
import unittest
from webtest import TestApp
import pecan
from pecan.tests import PecanTestCase
def test_overwrite_from_dict(self):
    from pecan import configuration
    configuration.set_config({'foo': 'bar'}, overwrite=True)
    assert dict(configuration._runtime_conf) == {'foo': 'bar'}