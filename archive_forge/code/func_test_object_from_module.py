import sys
import types
from testtools import TestCase
from testtools.matchers import (
from extras import (
def test_object_from_module(self):
    result = try_import('os.path.join')
    import os
    self.assertThat(result, Is(os.path.join))