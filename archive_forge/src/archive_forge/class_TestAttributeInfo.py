from oslo_utils import uuidutils
import testtools
from webob import exc
from neutron_lib.api import attributes
from neutron_lib.api import converters
from neutron_lib.api.definitions import network
from neutron_lib.api.definitions import port
from neutron_lib.api.definitions import subnet
from neutron_lib.api.definitions import subnetpool
from neutron_lib import constants
from neutron_lib import context
from neutron_lib import exceptions
from neutron_lib.tests import _base as base
class TestAttributeInfo(base.BaseTestCase):

    class _MyException(Exception):
        pass
    _EXC_CLS = _MyException
    _RESOURCE_NAME = 'thing'
    _RESOURCE_ATTRS = {'name': {}, 'type': {}}
    _RESOURCE_MAP = {_RESOURCE_NAME: _RESOURCE_ATTRS}
    _ATTRS_INSTANCE = attributes.AttributeInfo(_RESOURCE_MAP)

    def test_create_from_attribute_info_instance(self):
        cloned_attrs = attributes.AttributeInfo(TestAttributeInfo._ATTRS_INSTANCE)
        self.assertEqual(TestAttributeInfo._ATTRS_INSTANCE.attributes, cloned_attrs.attributes)

    def test_create_from_api_def(self):
        self.assertEqual(port.RESOURCE_ATTRIBUTE_MAP, attributes.AttributeInfo(port.RESOURCE_ATTRIBUTE_MAP).attributes)

    def _test_fill_default_value(self, attr_inst, expected, res_dict, check_allow_post=True):
        attr_inst.fill_post_defaults(res_dict, check_allow_post=check_allow_post)
        self.assertEqual(expected, res_dict)

    def test_fill_default_value_ok(self):
        attr_info = {'key': {'allow_post': True, 'default': constants.ATTR_NOT_SPECIFIED}}
        attr_inst = attributes.AttributeInfo(attr_info)
        self._test_fill_default_value(attr_inst, {'key': 'X'}, {'key': 'X'})
        self._test_fill_default_value(attr_inst, {'key': constants.ATTR_NOT_SPECIFIED}, {})

    def test_override_no_allow_post(self):
        attr_info = {'key': {'allow_post': False, 'default': constants.ATTR_NOT_SPECIFIED}}
        attr_inst = attributes.AttributeInfo(attr_info)
        self._test_fill_default_value(attr_inst, {'key': 'X'}, {'key': 'X'}, check_allow_post=False)

    def test_fill_no_default_value_allow_post(self):
        attr_info = {'key': {'allow_post': True}}
        attr_inst = attributes.AttributeInfo(attr_info)
        self._test_fill_default_value(attr_inst, {'key': 'X'}, {'key': 'X'})
        self.assertRaises(exceptions.InvalidInput, self._test_fill_default_value, attr_inst, {'key': 'X'}, {})
        self.assertRaises(self._EXC_CLS, attr_inst.fill_post_defaults, {}, self._EXC_CLS)

    def test_fill_no_default_value_no_allow_post(self):
        attr_info = {'key': {'allow_post': False}}
        attr_inst = attributes.AttributeInfo(attr_info)
        self.assertRaises(exceptions.InvalidInput, self._test_fill_default_value, attr_inst, {'key': 'X'}, {'key': 'X'})
        self._test_fill_default_value(attr_inst, {}, {})
        self.assertRaises(self._EXC_CLS, attr_inst.fill_post_defaults, {'key': 'X'}, self._EXC_CLS)

    def test_fill_none_overridden_by_default(self):
        attr_info = {'key': {'allow_post': True, 'default': 42, 'default_overrides_none': True}}
        attr_inst = attributes.AttributeInfo(attr_info)
        self._test_fill_default_value(attr_inst, {'key': 42}, {'key': None})

    def _test_convert_value(self, attr_inst, expected, res_dict):
        attr_inst.convert_values(res_dict)
        self.assertEqual(expected, res_dict)

    def test_convert_value(self):
        attr_info = {'key': {}}
        attr_inst = attributes.AttributeInfo(attr_info)
        self._test_convert_value(attr_inst, {'key': constants.ATTR_NOT_SPECIFIED}, {'key': constants.ATTR_NOT_SPECIFIED})
        self._test_convert_value(attr_inst, {'key': 'X'}, {'key': 'X'})
        self._test_convert_value(attr_inst, {'other_key': 'X'}, {'other_key': 'X'})
        attr_info = {'key': {'convert_to': converters.convert_to_int}}
        attr_inst = attributes.AttributeInfo(attr_info)
        self._test_convert_value(attr_inst, {'key': constants.ATTR_NOT_SPECIFIED}, {'key': constants.ATTR_NOT_SPECIFIED})
        self._test_convert_value(attr_inst, {'key': 1}, {'key': '1'})
        self._test_convert_value(attr_inst, {'key': 1}, {'key': 1})
        self.assertRaises(exceptions.InvalidInput, self._test_convert_value, attr_inst, {'key': 1}, {'key': 'a'})
        attr_info = {'key': {'validate': {'type:uuid': None}}}
        attr_inst = attributes.AttributeInfo(attr_info)
        self._test_convert_value(attr_inst, {'key': constants.ATTR_NOT_SPECIFIED}, {'key': constants.ATTR_NOT_SPECIFIED})
        uuid_str = '01234567-1234-1234-1234-1234567890ab'
        self._test_convert_value(attr_inst, {'key': uuid_str}, {'key': uuid_str})
        self.assertRaises(exceptions.InvalidInput, self._test_convert_value, attr_inst, {'key': 1}, {'key': 1})
        self.assertRaises(self._EXC_CLS, attr_inst.convert_values, {'key': 1}, self._EXC_CLS)

    def test_populate_project_id_admin_req(self):
        tenant_id_1 = uuidutils.generate_uuid()
        tenant_id_2 = uuidutils.generate_uuid()
        ctx = context.Context(user_id=None, tenant_id=tenant_id_1)
        res_dict = {'tenant_id': tenant_id_2}
        attr_inst = attributes.AttributeInfo({})
        self.assertRaises(exc.HTTPBadRequest, attr_inst.populate_project_id, ctx, res_dict, None)
        ctx.is_admin = True
        attr_inst.populate_project_id(ctx, res_dict, is_create=False)

    def test_populate_project_id_from_context(self):
        tenant_id = uuidutils.generate_uuid()
        ctx = context.Context(user_id=None, tenant_id=tenant_id)
        res_dict = {}
        attr_inst = attributes.AttributeInfo({})
        attr_inst.populate_project_id(ctx, res_dict, is_create=True)
        self.assertEqual({'tenant_id': ctx.tenant_id, 'project_id': ctx.tenant_id}, res_dict)

    def test_populate_project_id_mandatory_not_specified(self):
        tenant_id = uuidutils.generate_uuid()
        ctx = context.Context(user_id=None, tenant_id=tenant_id)
        res_dict = {}
        attr_info = {'tenant_id': {'allow_post': True}}
        ctx.tenant_id = None
        attr_inst = attributes.AttributeInfo(attr_info)
        self.assertRaises(exc.HTTPBadRequest, attr_inst.populate_project_id, ctx, res_dict, True)

    def test_populate_project_id_not_mandatory(self):
        ctx = context.Context(user_id=None)
        res_dict = {'name': 'test_port'}
        attr_inst = attributes.AttributeInfo({})
        ctx.tenant_id = None
        attr_inst.populate_project_id(ctx, res_dict, True)

    def test_verify_attributes_null(self):
        attributes.AttributeInfo({}).verify_attributes({})

    def test_verify_attributes_ok_with_project_id(self):
        attributes.AttributeInfo({'tenant_id': 'foo', 'project_id': 'foo'}).verify_attributes({'tenant_id': 'foo'})

    def test_verify_attributes_ok_subset(self):
        attributes.AttributeInfo({'attr1': 'foo', 'attr2': 'bar'}).verify_attributes({'attr1': 'foo'})

    def test_verify_attributes_unrecognized(self):
        with testtools.ExpectedException(exc.HTTPBadRequest) as bad_req:
            attributes.AttributeInfo({'attr1': 'foo'}).verify_attributes({'attr1': 'foo', 'attr2': 'bar'})
            self.assertEqual(bad_req.message, "Unrecognized attribute(s) 'attr2'")