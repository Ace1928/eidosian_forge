from unittest import mock
from unittest.mock import patch
import uuid
import glance_store
from oslo_config import cfg
from glance.common import exception
from glance.db.sqlalchemy import api as db_api
from glance import scrubber
from glance.tests import utils as test_utils
class ImagePager(object):

    def __init__(self, images, page_size=0):
        image_count = len(images)
        if page_size == 0 or page_size > image_count:
            page_size = image_count
        self.image_batches = []
        start = 0
        while start < image_count:
            self.image_batches.append(images[start:start + page_size])
            start += page_size
            if image_count - start < page_size:
                page_size = image_count - start

    def __call__(self):
        if len(self.image_batches) == 0:
            return []
        else:
            return self.image_batches.pop(0)