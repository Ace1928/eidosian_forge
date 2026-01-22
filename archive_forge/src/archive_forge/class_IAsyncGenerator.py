import sys
from abc import ABCMeta
from collections import OrderedDict
from collections import UserDict
from collections import UserList
from collections import UserString
from collections import abc
from zope.interface.common import ABCInterface
from zope.interface.common import optional
class IAsyncGenerator(IAsyncIterator):
    abc = _new_in_ver('AsyncGenerator', True)