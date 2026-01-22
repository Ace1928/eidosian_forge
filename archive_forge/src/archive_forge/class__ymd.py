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
class _ymd(list):

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.century_specified = False
        self.dstridx = None
        self.mstridx = None
        self.ystridx = None

    @property
    def has_year(self):
        return self.ystridx is not None

    @property
    def has_month(self):
        return self.mstridx is not None

    @property
    def has_day(self):
        return self.dstridx is not None

    def could_be_day(self, value):
        if self.has_day:
            return False
        elif not self.has_month:
            return 1 <= value <= 31
        elif not self.has_year:
            month = self[self.mstridx]
            return 1 <= value <= monthrange(2000, month)[1]
        else:
            month = self[self.mstridx]
            year = self[self.ystridx]
            return 1 <= value <= monthrange(year, month)[1]

    def append(self, val, label=None):
        if hasattr(val, '__len__'):
            if val.isdigit() and len(val) > 2:
                self.century_specified = True
                if label not in [None, 'Y']:
                    raise ValueError(label)
                label = 'Y'
        elif val > 100:
            self.century_specified = True
            if label not in [None, 'Y']:
                raise ValueError(label)
            label = 'Y'
        super(self.__class__, self).append(int(val))
        if label == 'M':
            if self.has_month:
                raise ValueError('Month is already set')
            self.mstridx = len(self) - 1
        elif label == 'D':
            if self.has_day:
                raise ValueError('Day is already set')
            self.dstridx = len(self) - 1
        elif label == 'Y':
            if self.has_year:
                raise ValueError('Year is already set')
            self.ystridx = len(self) - 1

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

    def resolve_ymd(self, yearfirst, dayfirst):
        len_ymd = len(self)
        year, month, day = (None, None, None)
        strids = (('y', self.ystridx), ('m', self.mstridx), ('d', self.dstridx))
        strids = {key: val for key, val in strids if val is not None}
        if len(self) == len(strids) > 0 or (len(self) == 3 and len(strids) == 2):
            return self._resolve_from_stridxs(strids)
        mstridx = self.mstridx
        if len_ymd > 3:
            raise ValueError('More than three YMD values')
        elif len_ymd == 1 or (mstridx is not None and len_ymd == 2):
            if mstridx is not None:
                month = self[mstridx]
                other = self[mstridx - 1]
            else:
                other = self[0]
            if len_ymd > 1 or mstridx is None:
                if other > 31:
                    year = other
                else:
                    day = other
        elif len_ymd == 2:
            if self[0] > 31:
                year, month = self
            elif self[1] > 31:
                month, year = self
            elif dayfirst and self[1] <= 12:
                day, month = self
            else:
                month, day = self
        elif len_ymd == 3:
            if mstridx == 0:
                if self[1] > 31:
                    month, year, day = self
                else:
                    month, day, year = self
            elif mstridx == 1:
                if self[0] > 31 or (yearfirst and self[2] <= 31):
                    year, month, day = self
                else:
                    day, month, year = self
            elif mstridx == 2:
                if self[1] > 31:
                    day, year, month = self
                else:
                    year, day, month = self
            elif self[0] > 31 or self.ystridx == 0 or (yearfirst and self[1] <= 12 and (self[2] <= 31)):
                if dayfirst and self[2] <= 12:
                    year, day, month = self
                else:
                    year, month, day = self
            elif self[0] > 12 or (dayfirst and self[1] <= 12):
                day, month, year = self
            else:
                month, day, year = self
        return (year, month, day)