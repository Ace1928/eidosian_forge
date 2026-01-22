from unittest import mock
from zunclient import api_versions
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import versions
class WrapsTestCase(utils.TestCase):

    def _get_obj_with_vers(self, vers):
        return mock.MagicMock(api_version=api_versions.APIVersion(vers))

    def _side_effect_of_vers_method(self, *args, **kwargs):
        m = mock.MagicMock(start_version=args[1], end_version=args[2])
        m.name = args[0]
        return m

    @mock.patch('zunclient.api_versions._get_function_name')
    @mock.patch('zunclient.api_versions.VersionedMethod')
    def test_end_version_is_none(self, mock_versioned_method, mock_name):
        func_name = 'foo'
        mock_name.return_value = func_name
        mock_versioned_method.side_effect = self._side_effect_of_vers_method

        @api_versions.wraps('2.2')
        def foo(*args, **kwargs):
            pass
        foo(self._get_obj_with_vers('2.4'))
        mock_versioned_method.assert_called_once_with(func_name, api_versions.APIVersion('2.2'), api_versions.APIVersion('2.latest'), mock.ANY)

    @mock.patch('zunclient.api_versions._get_function_name')
    @mock.patch('zunclient.api_versions.VersionedMethod')
    def test_start_and_end_version_are_presented(self, mock_versioned_method, mock_name):
        func_name = 'foo'
        mock_name.return_value = func_name
        mock_versioned_method.side_effect = self._side_effect_of_vers_method

        @api_versions.wraps('2.2', '2.6')
        def foo(*args, **kwargs):
            pass
        foo(self._get_obj_with_vers('2.4'))
        mock_versioned_method.assert_called_once_with(func_name, api_versions.APIVersion('2.2'), api_versions.APIVersion('2.6'), mock.ANY)

    @mock.patch('zunclient.api_versions._get_function_name')
    @mock.patch('zunclient.api_versions.VersionedMethod')
    def test_api_version_doesnt_match(self, mock_versioned_method, mock_name):
        func_name = 'foo'
        mock_name.return_value = func_name
        mock_versioned_method.side_effect = self._side_effect_of_vers_method

        @api_versions.wraps('2.2', '2.6')
        def foo(*args, **kwargs):
            pass
        self.assertRaises(exceptions.VersionNotFoundForAPIMethod, foo, self._get_obj_with_vers('2.1'))
        mock_versioned_method.assert_called_once_with(func_name, api_versions.APIVersion('2.2'), api_versions.APIVersion('2.6'), mock.ANY)

    def test_define_method_is_actually_called(self):
        checker = mock.MagicMock()

        @api_versions.wraps('2.2', '2.6')
        def some_func(*args, **kwargs):
            checker(*args, **kwargs)
        obj = self._get_obj_with_vers('2.4')
        some_args = ('arg_1', 'arg_2')
        some_kwargs = {'key1': 'value1', 'key2': 'value2'}
        some_func(obj, *some_args, **some_kwargs)
        checker.assert_called_once_with(*(obj,) + some_args, **some_kwargs)