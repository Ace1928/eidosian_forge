from unittest import mock
from zunclient import api_versions
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import versions
def test_version_comparisons(self):
    v1 = api_versions.APIVersion('2.0')
    v2 = api_versions.APIVersion('2.5')
    v3 = api_versions.APIVersion('5.23')
    v4 = api_versions.APIVersion('2.0')
    v_null = api_versions.APIVersion()
    self.assertTrue(v1 < v2)
    self.assertTrue(v3 > v2)
    self.assertTrue(v1 != v2)
    self.assertTrue(v1 == v4)
    self.assertTrue(v1 != v_null)
    self.assertTrue(v_null == v_null)
    self.assertRaises(TypeError, v1.__le__, '2.1')