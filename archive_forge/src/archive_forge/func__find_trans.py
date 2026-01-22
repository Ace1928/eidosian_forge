import bisect
import calendar
import collections
import functools
import re
import weakref
from datetime import datetime, timedelta, tzinfo
from . import _common, _tzpath
def _find_trans(self, dt):
    if dt is None:
        if self._fixed_offset:
            return self._tz_after
        else:
            return _NO_TTINFO
    ts = self._get_local_timestamp(dt)
    lt = self._trans_local[dt.fold]
    num_trans = len(lt)
    if num_trans and ts < lt[0]:
        return self._tti_before
    elif not num_trans or ts > lt[-1]:
        if isinstance(self._tz_after, _TZStr):
            return self._tz_after.get_trans_info(ts, dt.year, dt.fold)
        else:
            return self._tz_after
    else:
        idx = bisect.bisect_right(lt, ts) - 1
        assert idx >= 0
        return self._ttinfos[idx]