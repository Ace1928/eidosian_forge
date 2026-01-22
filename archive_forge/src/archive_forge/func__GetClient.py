import json
import os
import unittest
import six
from apitools.base.py import exceptions
import storage
def _GetClient():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = storage.StorageV1()
    return _CLIENT