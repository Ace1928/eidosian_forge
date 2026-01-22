from abc import ABCMeta, abstractmethod
from collections import namedtuple
from collections.abc import Mapping
import math
import select
import sys
class PollSelector(_PollLikeSelector):
    """Poll-based selector."""
    _selector_cls = select.poll
    _EVENT_READ = select.POLLIN
    _EVENT_WRITE = select.POLLOUT