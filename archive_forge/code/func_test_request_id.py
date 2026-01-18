from unittest import mock
from oslo_config import cfg
from glance import context
from glance.tests.unit import utils as unit_utils
from glance.tests import utils
def test_request_id(self):
    contexts = [context.RequestContext().request_id for _ in range(5)]
    self.assertEqual(5, len(set(contexts)))