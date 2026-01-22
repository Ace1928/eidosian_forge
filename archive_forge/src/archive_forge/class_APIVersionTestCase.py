from unittest import mock
from zunclient import api_versions
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import versions
class APIVersionTestCase(utils.TestCase):

    def test_valid_version_strings(self):

        def _test_string(version, exp_major, exp_minor):
            v = api_versions.APIVersion(version)
            self.assertEqual(v.ver_major, exp_major)
            self.assertEqual(v.ver_minor, exp_minor)
        _test_string('1.1', 1, 1)
        _test_string('2.10', 2, 10)
        _test_string('5.234', 5, 234)
        _test_string('12.5', 12, 5)
        _test_string('2.0', 2, 0)
        _test_string('2.200', 2, 200)

    def test_null_version(self):
        v = api_versions.APIVersion()
        self.assertTrue(v.is_null())

    def test_invalid_version_strings(self):
        self.assertRaises(exceptions.UnsupportedVersion, api_versions.APIVersion, '2')
        self.assertRaises(exceptions.UnsupportedVersion, api_versions.APIVersion, '200')
        self.assertRaises(exceptions.UnsupportedVersion, api_versions.APIVersion, '2.1.4')
        self.assertRaises(exceptions.UnsupportedVersion, api_versions.APIVersion, '200.23.66.3')
        self.assertRaises(exceptions.UnsupportedVersion, api_versions.APIVersion, '5 .3')
        self.assertRaises(exceptions.UnsupportedVersion, api_versions.APIVersion, '5. 3')
        self.assertRaises(exceptions.UnsupportedVersion, api_versions.APIVersion, '5.03')
        self.assertRaises(exceptions.UnsupportedVersion, api_versions.APIVersion, '02.1')
        self.assertRaises(exceptions.UnsupportedVersion, api_versions.APIVersion, '2.001')
        self.assertRaises(exceptions.UnsupportedVersion, api_versions.APIVersion, '')
        self.assertRaises(exceptions.UnsupportedVersion, api_versions.APIVersion, ' 2.1')
        self.assertRaises(exceptions.UnsupportedVersion, api_versions.APIVersion, '2.1 ')

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

    def test_get_string(self):
        v1_string = '3.23'
        v1 = api_versions.APIVersion(v1_string)
        self.assertEqual(v1_string, v1.get_string())
        self.assertRaises(ValueError, api_versions.APIVersion().get_string)