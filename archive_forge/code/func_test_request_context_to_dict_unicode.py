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
def test_request_context_to_dict_unicode(self):
    ctx_origin = {'username': 'mick', 'trustor_user_id': None, 'auth_token': '123', 'auth_token_info': {'123info': 'woop'}, 'is_admin': False, 'user': 'mick', 'password': 'foo', 'trust_id': None, 'global_request_id': None, 'show_deleted': False, 'roles': ['arole', 'notadmin'], 'tenant_id': '456tenant', 'project_id': '456tenant', 'user_id': u'Gāo', 'tenant': u'刘胜', 'project_name': u'刘胜', 'auth_url': 'http://xyz', 'aws_creds': 'blah', 'region_name': 'RegionOne', 'user_identity': u'Gāo 456tenant', 'user_domain_id': None, 'project_domain_id': None}
    ctx = context.RequestContext(auth_token=ctx_origin.get('auth_token'), username=ctx_origin.get('username'), password=ctx_origin.get('password'), aws_creds=ctx_origin.get('aws_creds'), project_name=ctx_origin.get('tenant'), project_id=ctx_origin.get('tenant_id'), user=ctx_origin.get('user_id'), auth_url=ctx_origin.get('auth_url'), roles=ctx_origin.get('roles'), show_deleted=ctx_origin.get('show_deleted'), is_admin=ctx_origin.get('is_admin'), auth_token_info=ctx_origin.get('auth_token_info'), trustor_user_id=ctx_origin.get('trustor_user_id'), trust_id=ctx_origin.get('trust_id'), region_name=ctx_origin.get('region_name'), user_domain_id=ctx_origin.get('user_domain_id'), project_domain_id=ctx_origin.get('project_domain_id'))
    ctx_dict = ctx.to_dict()
    del ctx_dict['request_id']
    self.assertEqual(ctx_origin, ctx_dict)