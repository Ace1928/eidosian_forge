from __future__ import unicode_literals
import datetime
import re
import string
import time
import warnings
from calendar import monthrange
from io import StringIO
import six
from six import integer_types, text_type
from decimal import Decimal
from warnings import warn
from .. import relativedelta
from .. import tz
def _resolve_from_stridxs(self, strids):
    """
        Try to resolve the identities of year/month/day elements using
        ystridx, mstridx, and dstridx, if enough of these are specified.
        """
    if len(self) == 3 and len(strids) == 2:
        missing = [x for x in range(3) if x not in strids.values()]
        key = [x for x in ['y', 'm', 'd'] if x not in strids]
        assert len(missing) == len(key) == 1
        key = key[0]
        val = missing[0]
        strids[key] = val
    assert len(self) == len(strids)
    out = {key: self[strids[key]] for key in strids}
    return (out.get('y'), out.get('m'), out.get('d'))