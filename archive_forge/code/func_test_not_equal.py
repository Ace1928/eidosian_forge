import pytest
from urllib3._collections import HTTPHeaderDict
from urllib3._collections import RecentlyUsedContainer as Container
from urllib3.exceptions import InvalidHeader
from urllib3.packages import six
def test_not_equal(self, d):
    b = HTTPHeaderDict(cookie='foo, bar')
    c = NonMappingHeaderContainer(cookie='foo, bar')
    assert not d != b
    assert not d != c
    assert d != 2