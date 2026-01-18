import os
import difflib
import unittest
import six
from apitools.gen import gen_client
from apitools.gen import test_utils
Like unittest.assertEqual with a diff in the exception message.