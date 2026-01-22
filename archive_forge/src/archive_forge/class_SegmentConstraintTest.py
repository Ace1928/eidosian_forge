from unittest import mock
from openstack import exceptions
from heat.engine.clients.os import openstacksdk
from heat.tests import common
from heat.tests import utils
class SegmentConstraintTest(common.HeatTestCase):

    def setUp(self):
        super(SegmentConstraintTest, self).setUp()
        self.ctx = utils.dummy_context()
        self.mock_find_segment = mock.Mock()
        self.ctx.clients.client_plugin('openstack').find_network_segment = self.mock_find_segment
        self.constraint = openstacksdk.SegmentConstraint()

    def test_validation(self):
        self.mock_find_segment.side_effect = ['seg1', exceptions.ResourceNotFound(), exceptions.DuplicateResource()]
        self.assertTrue(self.constraint.validate('foo', self.ctx))
        self.assertFalse(self.constraint.validate('bar', self.ctx))
        self.assertFalse(self.constraint.validate('baz', self.ctx))