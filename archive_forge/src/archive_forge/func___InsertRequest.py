import json
import os
import random
import string
import unittest
import six
from apitools.base.py import transfer
import storage
def __InsertRequest(self, filename):
    object_name = os.path.join(self._TESTDATA_PREFIX, filename)
    return storage.StorageObjectsInsertRequest(name=object_name, bucket=self._DEFAULT_BUCKET)