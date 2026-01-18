from unittest import mock
import futurist
import glance_store as store
from oslo_config import cfg
from taskflow.patterns import linear_flow
import glance.async_
from glance.async_.flows import api_image_import
import glance.tests.utils as test_utils
def test_set_threadpool_model_eventlet(self):
    glance.async_.set_threadpool_model('eventlet')
    self.assertEqual(glance.async_.EventletThreadPoolModel, glance.async_._THREADPOOL_MODEL)