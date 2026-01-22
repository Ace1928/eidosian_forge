import builtins
import errno
import hashlib
import io
import json
import os
import stat
from unittest import mock
import uuid
import fixtures
from oslo_utils.secretutils import md5
from oslo_utils import units
from glance_store._drivers import filesystem
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
def _store_image(self, in_metadata):
    expected_image_id = str(uuid.uuid4())
    expected_file_size = 10
    expected_file_contents = b'*' * expected_file_size
    image_file = io.BytesIO(expected_file_contents)
    self.store.FILESYSTEM_STORE_METADATA = in_metadata
    return self.store.add(expected_image_id, image_file, expected_file_size, self.hash_algo)