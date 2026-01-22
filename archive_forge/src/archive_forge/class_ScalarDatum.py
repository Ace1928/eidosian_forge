from typing import Collection, Sequence, Tuple, Union
import abc
import dataclasses
import enum
import numpy as np
class ScalarDatum:
    """A single datum in a scalar time series for a run and tag.

    Attributes:
      step: The global step at which this datum occurred; an integer. This
        is a unique key among data of this time series.
      wall_time: The real-world time at which this datum occurred, as
        `float` seconds since epoch.
      value: The scalar value for this datum; a `float`.
    """
    __slots__ = ('_step', '_wall_time', '_value')

    def __init__(self, step, wall_time, value):
        self._step = step
        self._wall_time = wall_time
        self._value = value

    @property
    def step(self):
        return self._step

    @property
    def wall_time(self):
        return self._wall_time

    @property
    def value(self):
        return self._value

    def __eq__(self, other):
        if not isinstance(other, ScalarDatum):
            return False
        if self._step != other._step:
            return False
        if self._wall_time != other._wall_time:
            return False
        if self._value != other._value:
            return False
        return True

    def __hash__(self):
        return hash((self._step, self._wall_time, self._value))

    def __repr__(self):
        return 'ScalarDatum(%s)' % ', '.join(('step=%r' % (self._step,), 'wall_time=%r' % (self._wall_time,), 'value=%r' % (self._value,)))