import http.client
import os
import sys
import time
import httplib2
from oslo_config import cfg
from oslo_serialization import jsonutils
from oslo_utils.fixture import uuidsentinel as uuids
from glance import context
import glance.db as db_api
from glance.tests import functional
from glance.tests.utils import execute
def _get_pending_delete_image(self, image_id):
    db_api.get_api()._FACADE = None
    image = db_api.get_api().image_get(self.admin_context, image_id)
    return image