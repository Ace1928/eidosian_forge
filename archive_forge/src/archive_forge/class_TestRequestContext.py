import os
from unittest import mock
from keystoneauth1 import loading as ks_loading
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_middleware import request_id
from oslo_policy import opts as policy_opts
from oslo_utils import importutils
import webob
from heat.common import context
from heat.common import exception
from heat.tests import common
class TestRequestContext(common.HeatTestCase):

    def setUp(self):
        self.ctx = {'username': 'mick', 'trustor_user_id': None, 'auth_token': '123', 'auth_token_info': {'123info': 'woop'}, 'is_admin': False, 'user': 'mick', 'password': 'foo', 'trust_id': None, 'global_request_id': None, 'show_deleted': False, 'roles': ['arole', 'notadmin'], 'tenant_id': '456tenant', 'user_id': 'fooUser', 'tenant': u'刘胜', 'auth_url': 'http://xyz', 'aws_creds': 'blah', 'region_name': 'RegionOne', 'user_identity': 'fooUser 456tenant', 'user_domain_id': None, 'project_domain_id': None}
        super(TestRequestContext, self).setUp()

    def test_request_context_init(self):
        ctx = context.RequestContext(auth_token=self.ctx.get('auth_token'), username=self.ctx.get('username'), password=self.ctx.get('password'), aws_creds=self.ctx.get('aws_creds'), project_name=self.ctx.get('tenant'), project_id=self.ctx.get('tenant_id'), user=self.ctx.get('user_id'), auth_url=self.ctx.get('auth_url'), roles=self.ctx.get('roles'), show_deleted=self.ctx.get('show_deleted'), is_admin=self.ctx.get('is_admin'), auth_token_info=self.ctx.get('auth_token_info'), trustor_user_id=self.ctx.get('trustor_user_id'), trust_id=self.ctx.get('trust_id'), region_name=self.ctx.get('region_name'), user_domain_id=self.ctx.get('user_domain_id'), project_domain_id=self.ctx.get('project_domain_id'))
        ctx_dict = ctx.to_dict()
        del ctx_dict['request_id']
        del ctx_dict['project_id']
        del ctx_dict['project_name']
        self.assertEqual(self.ctx, ctx_dict)

    def test_request_context_to_dict_unicode(self):
        ctx_origin = {'username': 'mick', 'trustor_user_id': None, 'auth_token': '123', 'auth_token_info': {'123info': 'woop'}, 'is_admin': False, 'user': 'mick', 'password': 'foo', 'trust_id': None, 'global_request_id': None, 'show_deleted': False, 'roles': ['arole', 'notadmin'], 'tenant_id': '456tenant', 'project_id': '456tenant', 'user_id': u'Gāo', 'tenant': u'刘胜', 'project_name': u'刘胜', 'auth_url': 'http://xyz', 'aws_creds': 'blah', 'region_name': 'RegionOne', 'user_identity': u'Gāo 456tenant', 'user_domain_id': None, 'project_domain_id': None}
        ctx = context.RequestContext(auth_token=ctx_origin.get('auth_token'), username=ctx_origin.get('username'), password=ctx_origin.get('password'), aws_creds=ctx_origin.get('aws_creds'), project_name=ctx_origin.get('tenant'), project_id=ctx_origin.get('tenant_id'), user=ctx_origin.get('user_id'), auth_url=ctx_origin.get('auth_url'), roles=ctx_origin.get('roles'), show_deleted=ctx_origin.get('show_deleted'), is_admin=ctx_origin.get('is_admin'), auth_token_info=ctx_origin.get('auth_token_info'), trustor_user_id=ctx_origin.get('trustor_user_id'), trust_id=ctx_origin.get('trust_id'), region_name=ctx_origin.get('region_name'), user_domain_id=ctx_origin.get('user_domain_id'), project_domain_id=ctx_origin.get('project_domain_id'))
        ctx_dict = ctx.to_dict()
        del ctx_dict['request_id']
        self.assertEqual(ctx_origin, ctx_dict)

    def test_request_context_from_dict(self):
        ctx = context.RequestContext.from_dict(self.ctx)
        ctx_dict = ctx.to_dict()
        del ctx_dict['request_id']
        del ctx_dict['project_id']
        del ctx_dict['project_name']
        self.assertEqual(self.ctx, ctx_dict)

    def test_request_context_update(self):
        ctx = context.RequestContext.from_dict(self.ctx)
        for k in self.ctx:
            if k == 'user_identity' or k == 'user_domain_id' or k == 'project_domain_id':
                continue
            if k == 'tenant' or k == 'user':
                continue
            self.assertEqual(self.ctx.get(k), ctx.to_dict().get(k))
            override = '%s_override' % k
            setattr(ctx, k, override)
            self.assertEqual(override, ctx.to_dict().get(k))

    def test_get_admin_context(self):
        ctx = context.get_admin_context()
        self.assertTrue(ctx.is_admin)
        self.assertFalse(ctx.show_deleted)

    def test_get_admin_context_show_deleted(self):
        ctx = context.get_admin_context(show_deleted=True)
        self.assertTrue(ctx.is_admin)
        self.assertTrue(ctx.show_deleted)

    def test_admin_context_policy_true(self):
        policy_check = 'heat.common.policy.Enforcer.check_is_admin'
        with mock.patch(policy_check) as pc:
            pc.return_value = True
            ctx = context.RequestContext(roles=['admin'])
            self.assertTrue(ctx.is_admin)

    def test_admin_context_policy_false(self):
        policy_check = 'heat.common.policy.Enforcer.check_is_admin'
        with mock.patch(policy_check) as pc:
            pc.return_value = False
            ctx = context.RequestContext(roles=['notadmin'])
            self.assertFalse(ctx.is_admin)

    def test_keystone_v3_endpoint_in_context(self):
        """Ensure that the context is the preferred source for the auth_uri."""
        cfg.CONF.set_override('auth_uri', 'http://xyz', group='clients_keystone')
        policy_check = 'heat.common.policy.Enforcer.check_is_admin'
        with mock.patch(policy_check) as pc:
            pc.return_value = False
            ctx = context.RequestContext(auth_url='http://example.com:5000/v2.0')
            self.assertEqual(ctx.keystone_v3_endpoint, 'http://example.com:5000/v3')

    def test_keystone_v3_endpoint_in_clients_keystone_config(self):
        """Ensure that the [clients_keystone] section is the preferred source.

        Ensure that the [clients_keystone] section of the configuration is
        the preferred source when the context does not have the auth_uri.
        """
        cfg.CONF.set_override('auth_uri', 'http://xyz', group='clients_keystone')
        policy_check = 'heat.common.policy.Enforcer.check_is_admin'
        with mock.patch(policy_check) as pc:
            pc.return_value = False
            with mock.patch('keystoneauth1.discover.Discover') as discover:

                class MockDiscover(object):

                    def url_for(self, endpoint):
                        return 'http://xyz/v3'
                discover.return_value = MockDiscover()
                ctx = context.RequestContext(auth_url=None)
                self.assertEqual(ctx.keystone_v3_endpoint, 'http://xyz/v3')

    def test_keystone_v3_endpoint_in_keystone_authtoken_config(self):
        """Ensure that the [keystone_authtoken] section is used.

        Ensure that the [keystone_authtoken] section of the configuration
        is used when the auth_uri is not defined in the context or the
        [clients_keystone] section.
        """
        importutils.import_module('keystonemiddleware.auth_token')
        try:
            cfg.CONF.set_override('www_authenticate_uri', 'http://abc/v2.0', group='keystone_authtoken')
        except cfg.NoSuchOptError:
            cfg.CONF.set_override('auth_uri', 'http://abc/v2.0', group='keystone_authtoken')
        policy_check = 'heat.common.policy.Enforcer.check_is_admin'
        with mock.patch(policy_check) as pc:
            pc.return_value = False
            ctx = context.RequestContext(auth_url=None)
            self.assertEqual(ctx.keystone_v3_endpoint, 'http://abc/v3')

    def test_keystone_v3_endpoint_not_set_in_config(self):
        """Ensure an exception is raised when the auth_uri cannot be obtained.

        Ensure an exception is raised when the auth_uri cannot be obtained
        from any source.
        """
        policy_check = 'heat.common.policy.Enforcer.check_is_admin'
        with mock.patch(policy_check) as pc:
            pc.return_value = False
            ctx = context.RequestContext(auth_url=None)
            self.assertRaises(exception.AuthorizationFailure, getattr, ctx, 'keystone_v3_endpoint')

    def test_get_trust_context_auth_plugin_unauthorized(self):
        self.ctx['trust_id'] = 'trust_id'
        ctx = context.RequestContext.from_dict(self.ctx)
        self.patchobject(ks_loading, 'load_auth_from_conf_options', return_value=None)
        self.assertRaises(exception.AuthorizationFailure, getattr, ctx, 'auth_plugin')

    def test_cache(self):
        ctx = context.RequestContext.from_dict(self.ctx)

        class Class1(object):
            pass

        class Class2(object):
            pass
        self.assertEqual(0, len(ctx._object_cache))
        cache1 = ctx.cache(Class1)
        self.assertIsInstance(cache1, Class1)
        self.assertEqual(1, len(ctx._object_cache))
        cache1a = ctx.cache(Class1)
        self.assertEqual(cache1, cache1a)
        self.assertEqual(1, len(ctx._object_cache))
        cache2 = ctx.cache(Class2)
        self.assertIsInstance(cache2, Class2)
        self.assertEqual(2, len(ctx._object_cache))