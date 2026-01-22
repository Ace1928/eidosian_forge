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
def _do_test_add_write_failure(self, errno, exception):
    filesystem.ChunkedFile.CHUNKSIZE = units.Ki
    image_id = str(uuid.uuid4())
    file_size = 5 * units.Ki
    file_contents = b'*' * file_size
    path = os.path.join(self.test_dir, image_id)
    image_file = io.BytesIO(file_contents)
    with mock.patch.object(builtins, 'open') as popen:
        e = IOError()
        e.errno = errno
        popen.side_effect = e
        self.assertRaises(exception, self.store.add, image_id, image_file, 0, self.hash_algo)
        self.assertFalse(os.path.exists(path))