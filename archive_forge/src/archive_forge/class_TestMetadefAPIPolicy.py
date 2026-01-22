from unittest import mock
import webob.exc
from glance.api.v2 import policy
from glance.common import exception
from glance.tests import utils
class TestMetadefAPIPolicy(APIPolicyBase):

    def setUp(self):
        super(TestMetadefAPIPolicy, self).setUp()
        self.enforcer = mock.MagicMock()
        self.md_resource = mock.MagicMock()
        self.context = mock.MagicMock()
        self.policy = policy.MetadefAPIPolicy(self.context, self.md_resource, enforcer=self.enforcer)

    def test_enforce(self):
        self.assertRaises(webob.exc.HTTPNotFound, super(TestMetadefAPIPolicy, self).test_enforce)

    def test_get_metadef_namespace(self):
        self.policy.get_metadef_namespace()
        self.enforcer.enforce.assert_called_once_with(self.context, 'get_metadef_namespace', mock.ANY)

    def test_get_metadef_namespaces(self):
        self.policy.get_metadef_namespaces()
        self.enforcer.enforce.assert_called_once_with(self.context, 'get_metadef_namespaces', mock.ANY)

    def test_add_metadef_namespace(self):
        self.policy.add_metadef_namespace()
        self.enforcer.enforce.assert_called_once_with(self.context, 'add_metadef_namespace', mock.ANY)

    def test_modify_metadef_namespace(self):
        self.policy.modify_metadef_namespace()
        self.enforcer.enforce.assert_called_once_with(self.context, 'modify_metadef_namespace', mock.ANY)

    def test_delete_metadef_namespace(self):
        self.policy.delete_metadef_namespace()
        self.enforcer.enforce.assert_called_once_with(self.context, 'delete_metadef_namespace', mock.ANY)

    def test_get_metadef_objects(self):
        self.policy.get_metadef_objects()
        self.enforcer.enforce.assert_called_once_with(self.context, 'get_metadef_objects', mock.ANY)

    def test_get_metadef_object(self):
        self.policy.get_metadef_object()
        self.enforcer.enforce.assert_called_once_with(self.context, 'get_metadef_object', mock.ANY)

    def test_add_metadef_object(self):
        self.policy.add_metadef_object()
        self.enforcer.enforce.assert_called_once_with(self.context, 'add_metadef_object', mock.ANY)

    def test_modify_metadef_object(self):
        self.policy.modify_metadef_object()
        self.enforcer.enforce.assert_called_once_with(self.context, 'modify_metadef_object', mock.ANY)

    def test_delete_metadef_object(self):
        self.policy.delete_metadef_object()
        self.enforcer.enforce.assert_called_once_with(self.context, 'delete_metadef_object', mock.ANY)

    def test_add_metadef_tag(self):
        self.policy.add_metadef_tag()
        self.enforcer.enforce.assert_called_once_with(self.context, 'add_metadef_tag', mock.ANY)

    def test_add_metadef_tags(self):
        self.policy.add_metadef_tags()
        self.enforcer.enforce.assert_called_once_with(self.context, 'add_metadef_tags', mock.ANY)

    def test_get_metadef_tags(self):
        self.policy.get_metadef_tags()
        self.enforcer.enforce.assert_called_once_with(self.context, 'get_metadef_tags', mock.ANY)

    def test_get_metadef_tag(self):
        self.policy.get_metadef_tag()
        self.enforcer.enforce.assert_called_once_with(self.context, 'get_metadef_tag', mock.ANY)

    def modify_metadef_tag(self):
        self.policy.modify_metadef_tag()
        self.enforcer.enforce.assert_called_once_with(self.context, 'modify_metadef_tag', mock.ANY)

    def test_delete_metadef_tags(self):
        self.policy.delete_metadef_tags()
        self.enforcer.enforce.assert_called_once_with(self.context, 'delete_metadef_tags', mock.ANY)

    def test_delete_metadef_tag(self):
        self.policy.delete_metadef_tag()
        self.enforcer.enforce.assert_called_once_with(self.context, 'delete_metadef_tag', mock.ANY)

    def test_add_metadef_property(self):
        self.policy.add_metadef_property()
        self.enforcer.enforce.assert_called_once_with(self.context, 'add_metadef_property', mock.ANY)

    def test_get_metadef_properties(self):
        self.policy.get_metadef_properties()
        self.enforcer.enforce.assert_called_once_with(self.context, 'get_metadef_properties', mock.ANY)

    def test_get_metadef_property(self):
        self.policy.get_metadef_property()
        self.enforcer.enforce.assert_called_once_with(self.context, 'get_metadef_property', mock.ANY)

    def test_modify_metadef_property(self):
        self.policy.modify_metadef_property()
        self.enforcer.enforce.assert_called_once_with(self.context, 'modify_metadef_property', mock.ANY)

    def test_remove_metadef_property(self):
        self.policy.remove_metadef_property()
        self.enforcer.enforce.assert_called_once_with(self.context, 'remove_metadef_property', mock.ANY)

    def test_add_metadef_resource_type_association(self):
        self.policy.add_metadef_resource_type_association()
        self.enforcer.enforce.assert_called_once_with(self.context, 'add_metadef_resource_type_association', mock.ANY)

    def test_list_metadef_resource_types(self):
        self.policy.list_metadef_resource_types()
        self.enforcer.enforce.assert_called_once_with(self.context, 'list_metadef_resource_types', mock.ANY)

    def test_enforce_exception_behavior(self):
        with mock.patch.object(self.policy.enforcer, 'enforce') as mock_enf:
            self.policy.modify_metadef_namespace()
            self.assertTrue(mock_enf.called)
            mock_enf.reset_mock()
            mock_enf.side_effect = exception.Forbidden
            self.assertRaises(webob.exc.HTTPNotFound, self.policy.modify_metadef_namespace)
            mock_enf.assert_has_calls([mock.call(mock.ANY, 'modify_metadef_namespace', mock.ANY), mock.call(mock.ANY, 'get_metadef_namespace', mock.ANY)])
            mock_enf.reset_mock()
            mock_enf.side_effect = [exception.Forbidden, lambda *a: None]
            self.assertRaises(webob.exc.HTTPForbidden, self.policy.modify_metadef_namespace)
            mock_enf.assert_has_calls([mock.call(mock.ANY, 'modify_metadef_namespace', mock.ANY), mock.call(mock.ANY, 'get_metadef_namespace', mock.ANY)])

    def test_get_metadef_resource_type(self):
        self.policy.get_metadef_resource_type()
        self.enforcer.enforce.assert_called_once_with(self.context, 'get_metadef_resource_type', mock.ANY)

    def test_remove_metadef_resource_type_association(self):
        self.policy.remove_metadef_resource_type_association()
        self.enforcer.enforce.assert_called_once_with(self.context, 'remove_metadef_resource_type_association', mock.ANY)