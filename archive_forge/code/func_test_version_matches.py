from unittest import mock
from zunclient import api_versions
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import versions
def test_version_matches(self):
    v1 = api_versions.APIVersion('2.0')
    v2 = api_versions.APIVersion('2.5')
    v3 = api_versions.APIVersion('2.45')
    v4 = api_versions.APIVersion('3.3')
    v5 = api_versions.APIVersion('3.23')
    v6 = api_versions.APIVersion('2.0')
    v7 = api_versions.APIVersion('3.3')
    v8 = api_versions.APIVersion('4.0')
    v_null = api_versions.APIVersion()
    self.assertTrue(v2.matches(v1, v3))
    self.assertTrue(v2.matches(v1, v_null))
    self.assertTrue(v1.matches(v6, v2))
    self.assertTrue(v4.matches(v2, v7))
    self.assertTrue(v4.matches(v_null, v7))
    self.assertTrue(v4.matches(v_null, v8))
    self.assertFalse(v1.matches(v2, v3))
    self.assertFalse(v5.matches(v2, v4))
    self.assertFalse(v2.matches(v3, v1))
    self.assertRaises(ValueError, v_null.matches, v1, v3)