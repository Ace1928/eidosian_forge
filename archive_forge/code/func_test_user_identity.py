from unittest import mock
from oslo_config import cfg
from glance import context
from glance.tests.unit import utils as unit_utils
from glance.tests import utils
def test_user_identity(self):
    ctx = context.RequestContext(user_id='user', project_id='tenant', domain_id='domain', user_domain_id='user-domain', project_domain_id='project-domain')
    self.assertEqual('user tenant domain user-domain project-domain', ctx.to_dict()['user_identity'])