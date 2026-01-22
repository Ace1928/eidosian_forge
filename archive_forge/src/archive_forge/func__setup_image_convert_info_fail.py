import json
import os
from unittest import mock
import glance_store
from oslo_concurrency import processutils
from oslo_config import cfg
import glance.async_.flows.api_image_import as import_flow
import glance.async_.flows.plugins.image_conversion as image_conversion
from glance.async_ import utils as async_utils
from glance.common import utils
from glance import domain
from glance import gateway
import glance.tests.utils as test_utils
def _setup_image_convert_info_fail(self):
    image_convert = image_conversion._ConvertImage(self.context, self.task.task_id, self.task_type, self.wrapper)
    self.task_repo.get.return_value = self.task
    image = mock.MagicMock(image_id=self.image_id, virtual_size=None, extra_properties={'os_glance_import_task': self.task.task_id}, disk_format='qcow2')
    self.img_repo.get.return_value = image
    return image_convert