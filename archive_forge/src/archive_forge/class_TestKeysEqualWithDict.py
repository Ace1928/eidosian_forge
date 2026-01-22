from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._dict import (
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestKeysEqualWithDict(TestKeysEqualWithList):
    matches_matcher = KeysEqual({'foo': 3, 'bar': 4})