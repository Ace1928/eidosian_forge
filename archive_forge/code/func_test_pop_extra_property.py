import sys
from unittest import mock
import urllib.error
from glance_store import exceptions as store_exceptions
from oslo_config import cfg
from oslo_utils import units
import taskflow
import glance.async_.flows.api_image_import as import_flow
from glance.common import exception
from glance.common.scripts.image_import import main as image_import
from glance import context
from glance.domain import ExtraProperties
from glance import gateway
import glance.tests.utils as test_utils
from cursive import exception as cursive_exception
def test_pop_extra_property(self):
    self.image.extra_properties = {'foo': '1', 'bar': 2}
    self.actions.pop_extra_property('foo')
    self.assertEqual({'bar': 2}, self.image.extra_properties)
    self.actions.pop_extra_property('baz')
    self.assertEqual({'bar': 2}, self.image.extra_properties)