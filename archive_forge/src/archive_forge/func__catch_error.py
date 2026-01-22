import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def _catch_error(self, ignore_errors=False):
    """
        Catch an exception and raise it unless asked to ignore it.

        :keyword  ignore_errors: If true, just return the error.  Otherwise,
                                 raise the error.
        :type     ignore_errors: ``bool``

        :return:  The exception that was raised.
        :rtype:   :class:`Exception`
        """
    e = sys.exc_info()[1]
    if ignore_errors:
        return e
    else:
        raise e