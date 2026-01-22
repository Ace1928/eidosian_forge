from unittest import mock
import uuid
from glanceclient import exc
from heat.engine.clients import client_exception as exception
from heat.engine.clients.os import glance
from heat.tests import common
from heat.tests import utils
class ImageConstraintTest(common.HeatTestCase):

    def setUp(self):
        super(ImageConstraintTest, self).setUp()
        self.ctx = utils.dummy_context()
        self.mock_find_image = mock.Mock()
        self.ctx.clients.client_plugin('glance').find_image_by_name_or_id = self.mock_find_image
        self.constraint = glance.ImageConstraint()

    def test_validation(self):
        self.mock_find_image.side_effect = ['id1', exception.EntityMatchNotFound(), exception.EntityUniqueMatchNotFound()]
        self.assertTrue(self.constraint.validate('foo', self.ctx))
        self.assertFalse(self.constraint.validate('bar', self.ctx))
        self.assertFalse(self.constraint.validate('baz', self.ctx))