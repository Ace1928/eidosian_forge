import pytest
from urllib3._collections import HTTPHeaderDict
from urllib3._collections import RecentlyUsedContainer as Container
from urllib3.exceptions import InvalidHeader
from urllib3.packages import six
def test_string_enforcement(self, d):
    with pytest.raises(Exception):
        d[3] = 5
    with pytest.raises(Exception):
        d.add(3, 4)
    with pytest.raises(Exception):
        del d[3]
    with pytest.raises(Exception):
        HTTPHeaderDict({3: 3})