import itertools
import json
import logging
from unittest import mock
from keystoneauth1 import adapter
import requests
from openstack import exceptions
from openstack import format
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
class TestComponent(base.TestCase):

    class ExampleComponent(resource._BaseComponent):
        key = '_example'

    def test_implementations(self):
        self.assertEqual('_body', resource.Body.key)
        self.assertEqual('_header', resource.Header.key)
        self.assertEqual('_uri', resource.URI.key)

    def test_creation(self):
        sot = resource._BaseComponent('name', type=int, default=1, alternate_id=True, aka='alias')
        self.assertEqual('name', sot.name)
        self.assertEqual(int, sot.type)
        self.assertEqual(1, sot.default)
        self.assertEqual('alias', sot.aka)
        self.assertTrue(sot.alternate_id)

    def test_get_no_instance(self):
        sot = resource._BaseComponent('test')
        result = sot.__get__(None, None)
        self.assertIs(sot, result)

    def test_get_name_None(self):
        name = 'name'

        class Parent:
            _example = {name: None}
        instance = Parent()
        sot = TestComponent.ExampleComponent(name, default=1)
        result = sot.__get__(instance, None)
        self.assertIsNone(result)

    def test_get_default(self):
        expected_result = 123

        class Parent:
            _example = {}
        instance = Parent()
        sot = TestComponent.ExampleComponent('name', type=dict, default=expected_result)
        result = sot.__get__(instance, None)
        self.assertEqual(expected_result, result)

    def test_get_name_untyped(self):
        name = 'name'
        expected_result = 123

        class Parent:
            _example = {name: expected_result}
        instance = Parent()
        sot = TestComponent.ExampleComponent('name')
        result = sot.__get__(instance, None)
        self.assertEqual(expected_result, result)

    def test_get_name_typed(self):
        name = 'name'
        value = '123'

        class Parent:
            _example = {name: value}
        instance = Parent()
        sot = TestComponent.ExampleComponent('name', type=int)
        result = sot.__get__(instance, None)
        self.assertEqual(int(value), result)

    def test_get_name_formatter(self):
        name = 'name'
        value = '123'
        expected_result = 'one hundred twenty three'

        class Parent:
            _example = {name: value}

        class FakeFormatter(format.Formatter):

            @classmethod
            def deserialize(cls, value):
                return expected_result
        instance = Parent()
        sot = TestComponent.ExampleComponent('name', type=FakeFormatter)
        result = sot.__get__(instance, None)
        self.assertEqual(expected_result, result)

    def test_set_name_untyped(self):
        name = 'name'
        expected_value = '123'

        class Parent:
            _example = {}
        instance = Parent()
        sot = TestComponent.ExampleComponent('name')
        sot.__set__(instance, expected_value)
        self.assertEqual(expected_value, instance._example[name])

    def test_set_name_typed(self):
        expected_value = '123'

        class Parent:
            _example = {}
        instance = Parent()

        class FakeType:
            calls = []

            def __init__(self, arg):
                FakeType.calls.append(arg)
        sot = TestComponent.ExampleComponent('name', type=FakeType)
        sot.__set__(instance, expected_value)
        self.assertEqual([expected_value], FakeType.calls)

    def test_set_name_formatter(self):
        expected_value = '123'

        class Parent:
            _example = {}
        instance = Parent()

        class FakeFormatter(format.Formatter):
            calls = []

            @classmethod
            def deserialize(cls, arg):
                FakeFormatter.calls.append(arg)
        sot = TestComponent.ExampleComponent('name', type=FakeFormatter)
        sot.__set__(instance, expected_value)
        self.assertEqual([expected_value], FakeFormatter.calls)

    def test_delete_name(self):
        name = 'name'
        expected_value = '123'

        class Parent:
            _example = {name: expected_value}
        instance = Parent()
        sot = TestComponent.ExampleComponent('name')
        sot.__delete__(instance)
        self.assertNotIn(name, instance._example)

    def test_delete_name_doesnt_exist(self):
        name = 'name'
        expected_value = '123'

        class Parent:
            _example = {'what': expected_value}
        instance = Parent()
        sot = TestComponent.ExampleComponent(name)
        sot.__delete__(instance)
        self.assertNotIn(name, instance._example)