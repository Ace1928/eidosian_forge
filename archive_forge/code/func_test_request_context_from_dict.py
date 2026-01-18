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
def test_request_context_from_dict(self):
    ctx = context.RequestContext.from_dict(self.ctx)
    ctx_dict = ctx.to_dict()
    del ctx_dict['request_id']
    del ctx_dict['project_id']
    del ctx_dict['project_name']
    self.assertEqual(self.ctx, ctx_dict)