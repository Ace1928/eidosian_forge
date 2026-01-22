import copy
import queue
from unittest import mock
from keystoneauth1 import session
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack import proxy
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
class TestProxyPrivate(base.TestCase):

    def setUp(self):
        super(TestProxyPrivate, self).setUp()

        def method(self, expected_type, value):
            return value
        self.sot = mock.Mock()
        self.sot.method = method
        self.session = mock.Mock()
        self.session._sdk_connection = self.cloud
        self.fake_proxy = proxy.Proxy(self.session)
        self.fake_proxy._connection = self.cloud

    def _test_correct(self, value):
        decorated = proxy._check_resource(strict=False)(self.sot.method)
        rv = decorated(self.sot, resource.Resource, value)
        self.assertEqual(value, rv)

    def test__check_resource_correct_resource(self):
        res = resource.Resource()
        self._test_correct(res)

    def test__check_resource_notstrict_id(self):
        self._test_correct('abc123-id')

    def test__check_resource_strict_id(self):
        decorated = proxy._check_resource(strict=True)(self.sot.method)
        self.assertRaisesRegex(ValueError, 'A Resource must be passed', decorated, self.sot, resource.Resource, 'this-is-not-a-resource')

    def test__check_resource_incorrect_resource(self):

        class OneType(resource.Resource):
            pass

        class AnotherType(resource.Resource):
            pass
        value = AnotherType()
        decorated = proxy._check_resource(strict=False)(self.sot.method)
        self.assertRaisesRegex(ValueError, 'Expected OneType but received AnotherType', decorated, self.sot, OneType, value)

    def test__get_uri_attribute_no_parent(self):

        class Child(resource.Resource):
            something = resource.Body('something')
        attr = 'something'
        value = 'nothing'
        child = Child(something=value)
        result = self.fake_proxy._get_uri_attribute(child, None, attr)
        self.assertEqual(value, result)

    def test__get_uri_attribute_with_parent(self):

        class Parent(resource.Resource):
            pass
        value = 'nothing'
        parent = Parent(id=value)
        result = self.fake_proxy._get_uri_attribute('child', parent, 'attr')
        self.assertEqual(value, result)

    def test__get_resource_new(self):
        value = 'hello'
        fake_type = mock.Mock(spec=resource.Resource)
        fake_type.new = mock.Mock(return_value=value)
        attrs = {'first': 'Brian', 'last': 'Curtin'}
        result = self.fake_proxy._get_resource(fake_type, None, **attrs)
        fake_type.new.assert_called_with(connection=self.cloud, **attrs)
        self.assertEqual(value, result)

    def test__get_resource_from_id(self):
        id = 'eye dee'
        value = 'hello'
        attrs = {'first': 'Brian', 'last': 'Curtin'}

        class Fake:
            call = {}

            @classmethod
            def new(cls, **kwargs):
                cls.call = kwargs
                return value
        result = self.fake_proxy._get_resource(Fake, id, **attrs)
        self.assertDictEqual(dict(id=id, connection=mock.ANY, **attrs), Fake.call)
        self.assertEqual(value, result)

    def test__get_resource_from_resource(self):
        res = mock.Mock(spec=resource.Resource)
        res._update = mock.Mock()
        attrs = {'first': 'Brian', 'last': 'Curtin'}
        result = self.fake_proxy._get_resource(resource.Resource, res, **attrs)
        res._update.assert_called_once_with(**attrs)
        self.assertEqual(result, res)

    def test__get_resource_from_munch(self):
        cls = mock.Mock()
        res = mock.Mock(spec=resource.Resource)
        res._update = mock.Mock()
        cls._from_munch.return_value = res
        m = utils.Munch(answer=42)
        attrs = {'first': 'Brian', 'last': 'Curtin'}
        result = self.fake_proxy._get_resource(cls, m, **attrs)
        cls._from_munch.assert_called_once_with(m, connection=self.cloud)
        res._update.assert_called_once_with(**attrs)
        self.assertEqual(result, res)