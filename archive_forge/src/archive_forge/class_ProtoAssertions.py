import difflib
import math
from ..compat import collections_abc
import six
from google.protobuf import descriptor
from google.protobuf import descriptor_pool
from google.protobuf import message
from google.protobuf import text_format
class ProtoAssertions(object):
    """Mix this into a googletest.TestCase class to get proto2 assertions.

  Usage:

  class SomeTestCase(compare.ProtoAssertions, googletest.TestCase):
    ...
    def testSomething(self):
      ...
      self.assertProtoEqual(a, b)

  See module-level definitions for method documentation.
  """

    def assertProtoEqual(self, *args, **kwargs):
        return assertProtoEqual(self, *args, **kwargs)