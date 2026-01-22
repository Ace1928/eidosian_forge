from abc import ABCMeta, abstractmethod
from collections import namedtuple
from collections.abc import Mapping
import math
import select
import sys
def _can_use(method):
    """Check if we can use the selector depending upon the
    operating system. """
    selector = getattr(select, method, None)
    if selector is None:
        return False
    try:
        selector_obj = selector()
        if method == 'poll':
            selector_obj.poll(0)
        else:
            selector_obj.close()
        return True
    except OSError:
        return False