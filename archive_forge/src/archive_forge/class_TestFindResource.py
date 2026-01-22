import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
class TestFindResource(test_utils.TestCase):

    def setUp(self):
        super(TestFindResource, self).setUp()
        self.name = 'legos'
        self.expected = mock.Mock()
        self.manager = mock.Mock()
        self.manager.resource_class = mock.Mock()
        self.manager.resource_class.__name__ = 'lego'

    def test_find_resource_get_int(self):
        self.manager.get = mock.Mock(return_value=self.expected)
        result = utils.find_resource(self.manager, 1)
        self.assertEqual(self.expected, result)
        self.manager.get.assert_called_with(1)

    def test_find_resource_get_int_string(self):
        self.manager.get = mock.Mock(return_value=self.expected)
        result = utils.find_resource(self.manager, '2')
        self.assertEqual(self.expected, result)
        self.manager.get.assert_called_with('2')

    def test_find_resource_get_name_and_domain(self):
        name = 'admin'
        domain_id = '30524568d64447fbb3fa8b7891c10dd6'
        side_effect = [Exception('Boom!'), self.expected]
        self.manager.get = mock.Mock(side_effect=side_effect)
        result = utils.find_resource(self.manager, name, domain_id=domain_id)
        self.assertEqual(self.expected, result)
        self.manager.get.assert_called_with(name, domain_id=domain_id)

    def test_find_resource_get_uuid(self):
        uuid = '9a0dc2a0-ad0d-11e3-a5e2-0800200c9a66'
        self.manager.get = mock.Mock(return_value=self.expected)
        result = utils.find_resource(self.manager, uuid)
        self.assertEqual(self.expected, result)
        self.manager.get.assert_called_with(uuid)

    def test_find_resource_get_whatever(self):
        self.manager.get = mock.Mock(return_value=self.expected)
        result = utils.find_resource(self.manager, 'whatever')
        self.assertEqual(self.expected, result)
        self.manager.get.assert_called_with('whatever')

    def test_find_resource_find(self):
        self.manager.get = mock.Mock(side_effect=Exception('Boom!'))
        self.manager.find = mock.Mock(return_value=self.expected)
        result = utils.find_resource(self.manager, self.name)
        self.assertEqual(self.expected, result)
        self.manager.get.assert_called_with(self.name)
        self.manager.find.assert_called_with(name=self.name)

    def test_find_resource_find_not_found(self):
        self.manager.get = mock.Mock(side_effect=Exception('Boom!'))
        self.manager.find = mock.Mock(side_effect=exceptions.NotFound(404, '2'))
        result = self.assertRaises(exceptions.CommandError, utils.find_resource, self.manager, self.name)
        self.assertEqual("No lego with a name or ID of 'legos' exists.", str(result))
        self.manager.get.assert_called_with(self.name)
        self.manager.find.assert_called_with(name=self.name)

    def test_find_resource_list_forbidden(self):
        self.manager.get = mock.Mock(side_effect=Exception('Boom!'))
        self.manager.find = mock.Mock(side_effect=Exception('Boom!'))
        self.manager.list = mock.Mock(side_effect=exceptions.Forbidden(403))
        self.assertRaises(exceptions.Forbidden, utils.find_resource, self.manager, self.name)
        self.manager.list.assert_called_with()

    def test_find_resource_find_no_unique(self):
        self.manager.get = mock.Mock(side_effect=Exception('Boom!'))
        self.manager.find = mock.Mock(side_effect=NoUniqueMatch())
        result = self.assertRaises(exceptions.CommandError, utils.find_resource, self.manager, self.name)
        self.assertEqual("More than one lego exists with the name 'legos'.", str(result))
        self.manager.get.assert_called_with(self.name)
        self.manager.find.assert_called_with(name=self.name)

    def test_find_resource_silly_resource(self):
        self.manager = mock.Mock()
        self.manager.get = mock.Mock(side_effect=Exception('Boom!'))
        self.manager.find = mock.Mock(side_effect=AttributeError("'Controller' object has no attribute 'find'"))
        silly_resource = FakeOddballResource(None, {'id': '12345', 'name': self.name}, loaded=True)
        self.manager.list = mock.Mock(return_value=[silly_resource])
        result = utils.find_resource(self.manager, self.name)
        self.assertEqual(silly_resource, result)
        self.manager.get.assert_called_with(self.name)
        self.manager.find.assert_called_with(name=self.name)

    def test_find_resource_silly_resource_not_found(self):
        self.manager = mock.Mock()
        self.manager.get = mock.Mock(side_effect=Exception('Boom!'))
        self.manager.find = mock.Mock(side_effect=AttributeError("'Controller' object has no attribute 'find'"))
        self.manager.list = mock.Mock(return_value=[])
        result = self.assertRaises(exceptions.CommandError, utils.find_resource, self.manager, self.name)
        self.assertEqual('Could not find resource legos', str(result))
        self.manager.get.assert_called_with(self.name)
        self.manager.find.assert_called_with(name=self.name)

    def test_find_resource_silly_resource_no_unique_match(self):
        self.manager = mock.Mock()
        self.manager.get = mock.Mock(side_effect=Exception('Boom!'))
        self.manager.find = mock.Mock(side_effect=AttributeError("'Controller' object has no attribute 'find'"))
        silly_resource = FakeOddballResource(None, {'id': '12345', 'name': self.name}, loaded=True)
        silly_resource_same = FakeOddballResource(None, {'id': 'abcde', 'name': self.name}, loaded=True)
        self.manager.list = mock.Mock(return_value=[silly_resource, silly_resource_same])
        result = self.assertRaises(exceptions.CommandError, utils.find_resource, self.manager, self.name)
        self.assertEqual("More than one resource exists with the name or ID 'legos'.", str(result))
        self.manager.get.assert_called_with(self.name)
        self.manager.find.assert_called_with(name=self.name)

    def test_format_dict(self):
        expected = "a='b', c='d', e='f'"
        self.assertEqual(expected, utils.format_dict({'a': 'b', 'c': 'd', 'e': 'f'}))
        self.assertEqual(expected, utils.format_dict({'e': 'f', 'c': 'd', 'a': 'b'}))
        self.assertIsNone(utils.format_dict(None))

    def test_format_dict_recursive(self):
        expected = "a='b', c.1='d', c.2=''"
        self.assertEqual(expected, utils.format_dict({'a': 'b', 'c': {'1': 'd', '2': ''}}))
        self.assertEqual(expected, utils.format_dict({'c': {'1': 'd', '2': ''}, 'a': 'b'}))
        self.assertIsNone(utils.format_dict(None))
        expected = "a1='A', a2.b1.c1='B', a2.b1.c2=, a2.b2='D'"
        self.assertEqual(expected, utils.format_dict({'a1': 'A', 'a2': {'b1': {'c1': 'B', 'c2': None}, 'b2': 'D'}}))
        self.assertEqual(expected, utils.format_dict({'a2': {'b1': {'c2': None, 'c1': 'B'}, 'b2': 'D'}, 'a1': 'A'}))

    def test_format_dict_of_list(self):
        expected = 'a=a1, a2; b=b1, b2; c=c1, c2; e='
        self.assertEqual(expected, utils.format_dict_of_list({'a': ['a2', 'a1'], 'b': ['b2', 'b1'], 'c': ['c1', 'c2'], 'd': None, 'e': []}))
        self.assertEqual(expected, utils.format_dict_of_list({'c': ['c1', 'c2'], 'a': ['a2', 'a1'], 'b': ['b2', 'b1'], 'e': []}))
        self.assertIsNone(utils.format_dict_of_list(None))

    def test_format_dict_of_list_with_separator(self):
        expected = 'a=a1, a2\nb=b1, b2\nc=c1, c2\ne='
        self.assertEqual(expected, utils.format_dict_of_list({'a': ['a2', 'a1'], 'b': ['b2', 'b1'], 'c': ['c1', 'c2'], 'd': None, 'e': []}, separator='\n'))
        self.assertEqual(expected, utils.format_dict_of_list({'c': ['c1', 'c2'], 'a': ['a2', 'a1'], 'b': ['b2', 'b1'], 'e': []}, separator='\n'))
        self.assertIsNone(utils.format_dict_of_list(None, separator='\n'))

    def test_format_list(self):
        expected = 'a, b, c'
        self.assertEqual(expected, utils.format_list(['a', 'b', 'c']))
        self.assertEqual(expected, utils.format_list(['c', 'b', 'a']))
        self.assertIsNone(utils.format_list(None))

    def test_format_list_of_dicts(self):
        expected = "a='b', c='d'\ne='f'"
        sorted_data = [{'a': 'b', 'c': 'd'}, {'e': 'f'}]
        unsorted_data = [{'c': 'd', 'a': 'b'}, {'e': 'f'}]
        self.assertEqual(expected, utils.format_list_of_dicts(sorted_data))
        self.assertEqual(expected, utils.format_list_of_dicts(unsorted_data))
        self.assertEqual('', utils.format_list_of_dicts([]))
        self.assertEqual('', utils.format_list_of_dicts([{}]))
        self.assertIsNone(utils.format_list_of_dicts(None))

    def test_format_list_separator(self):
        expected = 'a\nb\nc'
        actual_pre_sorted = utils.format_list(['a', 'b', 'c'], separator='\n')
        actual_unsorted = utils.format_list(['c', 'b', 'a'], separator='\n')
        self.assertEqual(expected, actual_pre_sorted)
        self.assertEqual(expected, actual_unsorted)