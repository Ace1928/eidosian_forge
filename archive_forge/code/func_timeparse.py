from __future__ import print_function, division, absolute_import
from collections import OrderedDict
from datetime import datetime, date, time, timedelta
from itertools import chain
import re
from textwrap import dedent
from types import MappingProxyType
from warnings import warn
from dateutil.parser import parse as dateparse
import numpy as np
from .dispatch import dispatch
from .coretypes import (int32, int64, float64, bool_, complex128, datetime_,
from .predicates import isdimension, isrecord
from .internal_utils import _toposort, groupby
from .util import subclasses
def timeparse(x, formats=('%H:%M:%S', '%H:%M:%S.%f')):
    msg = ''
    for format in formats:
        try:
            return datetime.strptime(x, format).time()
        except ValueError as e:
            msg = str(e)
    raise ValueError(msg)