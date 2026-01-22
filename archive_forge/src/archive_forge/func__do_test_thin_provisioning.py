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
def _do_test_thin_provisioning(self, content, size, truncate, write, thin):
    self.config(filesystem_store_chunk_size=units.Ki, filesystem_thin_provisioning=thin, group='glance_store')
    self.store.configure()
    image_file = io.BytesIO(content)
    image_id = str(uuid.uuid4())
    with mock.patch.object(builtins, 'open') as popen:
        self.store.add(image_id, image_file, size, self.hash_algo)
        write_count = popen.return_value.__enter__().write.call_count
        truncate_count = popen.return_value.__enter__().truncate.call_count
        self.assertEqual(write_count, write)
        self.assertEqual(truncate_count, truncate)