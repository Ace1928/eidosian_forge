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
def test_revert_updates_status_keys(self):
    img_repo = mock.MagicMock()
    task_repo = mock.MagicMock()
    wrapper = import_flow.ImportActionWrapper(img_repo, IMAGE_ID1, TASK_ID1)
    image_import = import_flow._ImportToStore(TASK_ID1, TASK_TYPE, task_repo, wrapper, 'http://url', 'store1', True, True)
    extra_properties = {'os_glance_importing_to_stores': 'store1,store2', 'os_glance_import_task': TASK_ID1}
    image = self.img_factory.new_image(image_id=UUID1, extra_properties=extra_properties)
    img_repo.get.return_value = image
    fail_key = 'os_glance_failed_import'
    pend_key = 'os_glance_importing_to_stores'
    image_import.revert(None)
    self.assertEqual('store2', image.extra_properties[pend_key])
    try:
        raise Exception('foo')
    except Exception:
        fake_exc_info = sys.exc_info()
    extra_properties = {'os_glance_importing_to_stores': 'store1,store2'}
    image_import.revert(taskflow.types.failure.Failure(fake_exc_info))
    self.assertEqual('store2', image.extra_properties[pend_key])
    self.assertEqual('store1', image.extra_properties[fail_key])