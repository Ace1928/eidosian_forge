import abc
import collections
import contextlib
import functools
import time
import enum
from oslo_utils import timeutils
from oslo_utils import uuidutils
import tenacity
from taskflow import exceptions as excp
from taskflow import states
from taskflow.types import notifier
from taskflow.utils import iter_utils
class JobPriority(enum.Enum):
    """Enum of job priorities (modeled after hadoop job priorities)."""
    VERY_HIGH = 'VERY_HIGH'
    HIGH = 'HIGH'
    NORMAL = 'NORMAL'
    LOW = 'LOW'
    VERY_LOW = 'VERY_LOW'

    @classmethod
    def convert(cls, value):
        if isinstance(value, cls):
            return value
        try:
            return cls(value.upper())
        except (ValueError, AttributeError):
            valids = [cls.VERY_HIGH, cls.HIGH, cls.NORMAL, cls.LOW, cls.VERY_LOW]
            valids = [p.value for p in valids]
            raise ValueError("'%s' is not a valid priority, valid priorities are %s" % (value, valids))

    @classmethod
    def reorder(cls, *values):
        """Reorders (priority, value) tuples -> priority ordered values."""
        if len(values) == 0:
            raise ValueError('At least one (priority, value) pair is required')
        elif len(values) == 1:
            v1 = values[0]
            p1 = cls.convert(v1[0])
            return v1[1]
        else:
            priority_ordering = (cls.VERY_HIGH, cls.HIGH, cls.NORMAL, cls.LOW, cls.VERY_LOW)
            if len(values) == 2:
                v1 = values[0]
                v2 = values[1]
                p1 = cls.convert(v1[0])
                p2 = cls.convert(v2[0])
                p1_i = priority_ordering.index(p1)
                p2_i = priority_ordering.index(p2)
                if p1_i <= p2_i:
                    return (v1[1], v2[1])
                else:
                    return (v2[1], v1[1])
            else:
                buckets = collections.defaultdict(list)
                for p, v in values:
                    p = cls.convert(p)
                    buckets[p].append(v)
                values = []
                for p in priority_ordering:
                    values.extend(buckets[p])
                return tuple(values)