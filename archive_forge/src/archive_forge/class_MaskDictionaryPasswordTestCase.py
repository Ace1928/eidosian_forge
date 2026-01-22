import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
class MaskDictionaryPasswordTestCase(test_base.BaseTestCase):

    def test_dictionary(self):
        payload = {'password': 'TL0EfN33'}
        expected = {'password': '***'}
        self.assertEqual(expected, strutils.mask_dict_password(payload))
        payload = {'password': 'TL0Ef"N33'}
        expected = {'password': '***'}
        self.assertEqual(expected, strutils.mask_dict_password(payload))
        payload = {'user': 'admin', 'password': 'TL0EfN33'}
        expected = {'user': 'admin', 'password': '***'}
        self.assertEqual(expected, strutils.mask_dict_password(payload))
        payload = {'strval': 'somestring', 'dictval': {'user': 'admin', 'password': 'TL0EfN33'}}
        expected = {'strval': 'somestring', 'dictval': {'user': 'admin', 'password': '***'}}
        self.assertEqual(expected, strutils.mask_dict_password(payload))
        payload = {'strval': '--password abc', 'dont_change': 'this is fine', 'dictval': {'user': 'admin', 'password': b'TL0EfN33'}}
        expected = {'strval': '--password ***', 'dont_change': 'this is fine', 'dictval': {'user': 'admin', 'password': '***'}}
        self.assertEqual(expected, strutils.mask_dict_password(payload))
        payload = {'ipmi_password': 'KeDrahishvowphyecMornEm0or('}
        expected = {'ipmi_password': '***'}
        self.assertEqual(expected, strutils.mask_dict_password(payload))
        payload = {'passwords': {'KeystoneFernetKey1': 'c5FijjS'}}
        expected = {'passwords': {'KeystoneFernetKey1': '***'}}
        self.assertEqual(expected, strutils.mask_dict_password(payload))
        payload = {'passwords': {'keystonecredential0': 'c5FijjS'}}
        expected = {'passwords': {'keystonecredential0': '***'}}
        self.assertEqual(expected, strutils.mask_dict_password(payload))

    def test_do_no_harm(self):
        payload = {}
        expected = {}
        self.assertEqual(expected, strutils.mask_dict_password(payload))
        payload = {'somekey': 'somevalue', 'anotherkey': 'anothervalue'}
        expected = {'somekey': 'somevalue', 'anotherkey': 'anothervalue'}
        self.assertEqual(expected, strutils.mask_dict_password(payload))

    def test_do_an_int(self):
        payload = {}
        payload[1] = 2
        expected = payload.copy()
        self.assertEqual(expected, strutils.mask_dict_password(payload))

    def test_mask_values(self):
        payload = {'somekey': 'test = cmd --password myé\x80\x80pass'}
        expected = {'somekey': 'test = cmd --password ***'}
        self.assertEqual(expected, strutils.mask_dict_password(payload))

    def test_other_non_str_values(self):
        payload = {'password': 'DK0PK1AK3', 'bool': True, 'dict': {'cat': 'meow', 'password': '*aa38skdjf'}, 'float': 0.1, 'int': 123, 'list': [1, 2], 'none': None, 'str': 'foo'}
        expected = {'password': '***', 'bool': True, 'dict': {'cat': 'meow', 'password': '***'}, 'float': 0.1, 'int': 123, 'list': [1, 2], 'none': None, 'str': 'foo'}
        self.assertEqual(expected, strutils.mask_dict_password(payload))

    def test_argument_untouched(self):
        """Make sure that the argument passed in is not modified"""
        payload = {'password': 'DK0PK1AK3', 'bool': True, 'dict': {'cat': 'meow', 'password': '*aa38skdjf'}, 'float': 0.1, 'int': 123, 'list': [1, 2], 'none': None, 'str': 'foo'}
        pristine = copy.deepcopy(payload)
        strutils.mask_dict_password(payload)
        self.assertEqual(pristine, payload)

    def test_non_dict(self):
        expected = {'password': '***', 'foo': 'bar'}
        payload = TestMapping()
        self.assertEqual(expected, strutils.mask_dict_password(payload))

    def test_nested_non_dict(self):
        expected = {'nested': {'password': '***', 'foo': 'bar'}}
        payload = NestedMapping()
        self.assertEqual(expected, strutils.mask_dict_password(payload))