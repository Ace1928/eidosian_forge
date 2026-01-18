from __future__ import print_function, division, absolute_import
from .dispatch import dispatch
from .coretypes import (
from .predicates import isdimension
from .util import dshape
from datetime import date, time, datetime
import numpy as np
@validate.register(String, str)
@validate.register(Time, time)
@validate.register(Date, date)
@validate.register(DateTime, datetime)
def validate_always_true(schema, value):
    return True