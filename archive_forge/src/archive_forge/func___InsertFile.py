import json
import os
import random
import string
import unittest
import six
from apitools.base.py import transfer
import storage
def __InsertFile(self, filename, request=None):
    if request is None:
        request = self.__InsertRequest(filename)
    response = self.__client.objects.Insert(request, upload=self.__upload)
    self.assertIsNotNone(response)
    self.__files.append(filename)
    return response