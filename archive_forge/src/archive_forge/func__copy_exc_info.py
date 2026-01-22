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
def _copy_exc_info(exc_info):
    if exc_info is None:
        return None
    exc_type, exc_value, tb = exc_info
    return (exc_type, copy.copy(exc_value), tb)