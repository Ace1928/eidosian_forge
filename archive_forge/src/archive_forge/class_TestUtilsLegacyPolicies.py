from unittest import mock
from oslo_config import cfg
from oslo_db.sqlalchemy import models
import sqlalchemy as sa
from sqlalchemy.ext import declarative
from sqlalchemy import orm
from neutron_lib.api import attributes
from neutron_lib import context
from neutron_lib.db import utils
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
class TestUtilsLegacyPolicies(base.BaseTestCase):

    def setUp(self):
        cfg.CONF.set_override('enforce_scope', False, group='oslo_policy')
        super().setUp()

    def test_get_sort_dirs(self):
        sorts = [(1, True), (2, False), (3, True)]
        self.assertEqual(['asc', 'desc', 'asc'], utils.get_sort_dirs(sorts))

    def test_get_sort_dirs_reversed(self):
        sorts = [(1, True), (2, False), (3, True)]
        self.assertEqual(['desc', 'asc', 'desc'], utils.get_sort_dirs(sorts, page_reverse=True))

    def test_get_and_validate_sort_keys(self):
        sorts = [('name', False), ('status', True)]
        self.assertEqual(['name', 'status'], utils.get_and_validate_sort_keys(sorts, FakePort))

    def test_get_and_validate_sort_keys_bad_key_fails(self):
        sorts = [('master', True)]
        self.assertRaises(n_exc.BadRequest, utils.get_and_validate_sort_keys, sorts, FakePort)

    def test_get_and_validate_sort_keys_by_relationship_fails(self):
        sorts = [('gw_port', True)]
        self.assertRaises(n_exc.BadRequest, utils.get_and_validate_sort_keys, sorts, FakeRouter)

    def test_get_marker_obj(self):
        plugin = mock.Mock()
        plugin._get_myr.return_value = 'obj'
        obj = utils.get_marker_obj(plugin, 'ctx', 'myr', 10, mock.ANY)
        self.assertEqual('obj', obj)
        plugin._get_myr.assert_called_once_with('ctx', mock.ANY)

    def test_get_marker_obj_no_limit_and_marker(self):
        self.assertIsNone(utils.get_marker_obj(mock.Mock(), 'ctx', 'myr', 0, mock.ANY))
        self.assertIsNone(utils.get_marker_obj(mock.Mock(), 'ctx', 'myr', 10, None))

    @mock.patch.object(attributes, 'populate_project_info')
    def test_resource_fields(self, mock_populate):
        r = {'name': 'n', 'id': '1', 'desc': None}
        utils.resource_fields(r, ['name'])
        mock_populate.assert_called_once_with({'name': 'n'})

    def test_model_query_scope_is_project_admin(self):
        ctx = context.Context(project_id='some project', is_admin=True, is_advsvc=False)
        model = mock.Mock(project_id='project')
        self.assertFalse(utils.model_query_scope_is_project(ctx, model))
        del model.project_id
        self.assertFalse(utils.model_query_scope_is_project(ctx, model))

    def test_model_query_scope_is_project_advsvc(self):
        ctx = context.Context(project_id='some project', is_admin=False, is_advsvc=True)
        model = mock.Mock(project_id='project')
        self.assertFalse(utils.model_query_scope_is_project(ctx, model))
        del model.project_id
        self.assertFalse(utils.model_query_scope_is_project(ctx, model))

    def test_model_query_scope_is_project_service_role(self):
        ctx = context.Context(project_id='some project', is_admin=False, roles=['service'])
        model = mock.Mock(project_id='project')
        self.assertFalse(utils.model_query_scope_is_project(ctx, model))
        del model.project_id
        self.assertFalse(utils.model_query_scope_is_project(ctx, model))

    def test_model_query_scope_is_project_regular_user(self):
        ctx = context.Context(project_id='some project', is_admin=False, is_advsvc=False)
        model = mock.Mock(project_id='project')
        self.assertTrue(utils.model_query_scope_is_project(ctx, model))
        del model.project_id
        self.assertFalse(utils.model_query_scope_is_project(ctx, model))

    def test_model_query_scope_is_project_system_scope(self):
        ctx = context.Context(system_scope='all')
        model = mock.Mock(project_id='project')
        self.assertTrue(utils.model_query_scope_is_project(ctx, model))
        del model.project_id
        self.assertFalse(utils.model_query_scope_is_project(ctx, model))