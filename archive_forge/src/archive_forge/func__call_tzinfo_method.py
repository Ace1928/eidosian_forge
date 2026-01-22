from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future.builtins import str
from future.builtins import bytes
from future.builtins import map
from future.builtins import round
from future.builtins import int
from future.builtins import object
from future.utils import native_str, PY2
import time as _time
import math as _math
def _call_tzinfo_method(tzinfo, methname, tzinfoarg):
    if tzinfo is None:
        return None
    return getattr(tzinfo, methname)(tzinfoarg)