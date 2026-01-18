import io
import json
import os
from unittest import mock
import urllib
import glance_store
from oslo_concurrency import processutils as putils
from oslo_config import cfg
from taskflow import task
from taskflow.types import failure
import glance.async_.flows.base_import as import_flow
from glance.async_ import taskflow_executor
from glance.async_ import utils as async_utils
from glance.common.scripts.image_import import main as image_import
from glance.common.scripts import utils as script_utils
from glance.common import utils
from glance import context
from glance import domain
from glance import gateway
import glance.tests.utils as test_utils
def test_save_image(self):
    save_image = import_flow._SaveImage(self.task.task_id, self.task_type, self.img_repo)
    with mock.patch.object(self.img_repo, 'get') as get_mock:
        image_id = mock.sentinel.image_id
        image = mock.MagicMock(image_id=image_id, status='saving')
        get_mock.return_value = image
        with mock.patch.object(self.img_repo, 'save') as save_mock:
            save_image.execute(image.image_id)
            get_mock.assert_called_once_with(image_id)
            save_mock.assert_called_once_with(image)
            self.assertEqual('active', image.status)