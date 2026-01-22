from typing import Collection, Sequence, Tuple, Union
import abc
import dataclasses
import enum
import numpy as np
class BlobSequenceDatum:
    """A single datum in a blob sequence time series for a run and tag.

    Attributes:
      step: The global step at which this datum occurred; an integer. This is a
        unique key among data of this time series.
      wall_time: The real-world time at which this datum occurred, as `float`
        seconds since epoch.
      values: A tuple of `BlobReference` objects, providing access to elements of
        this sequence.
    """
    __slots__ = ('_step', '_wall_time', '_values')

    def __init__(self, step, wall_time, values):
        self._step = step
        self._wall_time = wall_time
        self._values = values

    @property
    def step(self):
        return self._step

    @property
    def wall_time(self):
        return self._wall_time

    @property
    def values(self):
        return self._values

    def __eq__(self, other):
        if not isinstance(other, BlobSequenceDatum):
            return False
        if self._step != other._step:
            return False
        if self._wall_time != other._wall_time:
            return False
        if self._values != other._values:
            return False
        return True

    def __hash__(self):
        return hash((self._step, self._wall_time, self._values))

    def __repr__(self):
        return 'BlobSequenceDatum(%s)' % ', '.join(('step=%r' % (self._step,), 'wall_time=%r' % (self._wall_time,), 'values=%r' % (self._values,)))