import errno
import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
from unittest import mock
import uuid
import yaml
from oslotest import base as test_base
from oslo_utils import fileutils
def test_read_all(self):
    res = fileutils.write_to_tempfile(self.content)
    self.assertTrue(os.path.exists(res))
    out, unread_bytes = fileutils.last_bytes(res, 1000)
    self.assertEqual(b'1234567890', out)
    self.assertEqual(0, unread_bytes)