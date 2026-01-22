import json
import os
import unittest
import six
from apitools.base.py import exceptions
import storage
@classmethod
def __GetRequest(cls, filename):
    object_name = os.path.join(cls._TESTDATA_PREFIX, filename)
    return storage.StorageObjectsGetRequest(bucket=cls._DEFAULT_BUCKET, object=object_name)