import collections
import copy
import io
import os
import sys
import traceback
from oslo_utils import encodeutils
from oslo_utils import reflection
from taskflow import exceptions as exc
from taskflow.utils import iter_utils
from taskflow.utils import schema_utils as su
def _are_equal_exc_info_tuples(ei1, ei2):
    if ei1 == ei2:
        return True
    if ei1 is None or ei2 is None:
        return False
    if ei1[0] is not ei2[0]:
        return False
    if not all((type(ei1[1]) == type(ei2[1]), _exception_message(ei1[1]) == _exception_message(ei2[1]), repr(ei1[1]) == repr(ei2[1]))):
        return False
    if ei1[2] == ei2[2]:
        return True
    tb1 = traceback.format_tb(ei1[2])
    tb2 = traceback.format_tb(ei2[2])
    return tb1 == tb2