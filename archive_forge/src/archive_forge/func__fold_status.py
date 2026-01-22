from six import PY2
from functools import wraps
from datetime import datetime, timedelta, tzinfo
def _fold_status(self, dt_utc, dt_wall):
    """
        Determine the fold status of a "wall" datetime, given a representation
        of the same datetime as a (naive) UTC datetime. This is calculated based
        on the assumption that ``dt.utcoffset() - dt.dst()`` is constant for all
        datetimes, and that this offset is the actual number of hours separating
        ``dt_utc`` and ``dt_wall``.

        :param dt_utc:
            Representation of the datetime as UTC

        :param dt_wall:
            Representation of the datetime as "wall time". This parameter must
            either have a `fold` attribute or have a fold-naive
            :class:`datetime.tzinfo` attached, otherwise the calculation may
            fail.
        """
    if self.is_ambiguous(dt_wall):
        delta_wall = dt_wall - dt_utc
        _fold = int(delta_wall == dt_utc.utcoffset() - dt_utc.dst())
    else:
        _fold = 0
    return _fold