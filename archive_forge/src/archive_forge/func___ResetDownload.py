import json
import os
import unittest
import six
from apitools.base.py import exceptions
import storage
def __ResetDownload(self, auto_transfer=False):
    self.__buffer = six.StringIO()
    self.__download = storage.Download.FromStream(self.__buffer, auto_transfer=auto_transfer)