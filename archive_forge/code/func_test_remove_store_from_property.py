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
@mock.patch('glance.async_.flows.api_image_import.image_import')
def test_remove_store_from_property(self, mock_import):
    img_repo = mock.MagicMock()
    task_repo = mock.MagicMock()
    wrapper = import_flow.ImportActionWrapper(img_repo, IMAGE_ID1, TASK_ID1)
    image_import = import_flow._ImportToStore(TASK_ID1, TASK_TYPE, task_repo, wrapper, 'http://url', 'store1', True, True)
    extra_properties = {'os_glance_importing_to_stores': 'store1,store2', 'os_glance_import_task': TASK_ID1}
    image = self.img_factory.new_image(image_id=UUID1, extra_properties=extra_properties)
    img_repo.get.return_value = image
    image_import.execute()
    self.assertEqual(image.extra_properties['os_glance_importing_to_stores'], 'store2')