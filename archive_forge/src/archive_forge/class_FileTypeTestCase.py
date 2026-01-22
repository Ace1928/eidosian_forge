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
class FileTypeTestCase(test_base.BaseTestCase):
    """Test the is_yaml() and is_json() utility methods."""

    def setUp(self):
        super(FileTypeTestCase, self).setUp()
        data = {'name': 'test', 'website': 'example.com'}
        temp_dir = tempfile.mkdtemp()
        self.json_file = tempfile.mktemp(dir=temp_dir)
        self.yaml_file = tempfile.mktemp(dir=temp_dir)
        with open(self.json_file, 'w') as fh:
            json.dump(data, fh)
        with open(self.yaml_file, 'w') as fh:
            yaml.dump(data, fh)

    def test_is_json(self):
        self.assertTrue(fileutils.is_json(self.json_file))
        self.assertFalse(fileutils.is_json(self.yaml_file))

    def test_is_yaml(self):
        self.assertTrue(fileutils.is_yaml(self.yaml_file))
        self.assertFalse(fileutils.is_yaml(self.json_file))