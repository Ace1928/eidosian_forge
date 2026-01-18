import datetime
from unittest import mock
import uuid
from oslo_config import cfg
import oslo_utils.importutils
import glance.async_
from glance.async_ import taskflow_executor
from glance.common import exception
from glance.common import timeutils
from glance import domain
import glance.tests.utils as test_utils
def test_tags_always_a_set(self):
    self.image.tags = ['a', 'b', 'c']
    self.assertEqual(set(['a', 'b', 'c']), self.image.tags)