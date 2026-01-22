from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
def _commonprefix(strings):
    if not strings:
        return ''
    else:
        s1 = min(strings)
        s2 = max(strings)
        for i, c in enumerate(s1):
            if c != s2[i]:
                return s1[:i]
        return s1