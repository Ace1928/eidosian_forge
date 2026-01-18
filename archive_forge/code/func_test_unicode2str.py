import random
import sys
from monty.string import remove_non_ascii, unicode2str
def test_unicode2str():
    if sys.version_info.major < 3:
        assert type(unicode2str('a')) == str
    else:
        assert type(unicode2str('a')) == str