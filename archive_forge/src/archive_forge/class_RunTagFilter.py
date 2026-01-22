from typing import Collection, Sequence, Tuple, Union
import abc
import dataclasses
import enum
import numpy as np
class RunTagFilter:
    """Filters data by run and tag names."""

    def __init__(self, runs=None, tags=None):
        """Construct a `RunTagFilter`.

        A time series passes this filter if both its run *and* its tag are
        included in the corresponding whitelists.

        Order and multiplicity are ignored; `runs` and `tags` are treated as
        sets.

        Args:
          runs: Collection of run names, as strings, or `None` to admit all
            runs.
          tags: Collection of tag names, as strings, or `None` to admit all
            tags.
        """
        self._runs = self._parse_optional_string_set('runs', runs)
        self._tags = self._parse_optional_string_set('tags', tags)

    def _parse_optional_string_set(self, name, value):
        if value is None:
            return None
        if isinstance(value, str):
            raise TypeError('%s: expected `None` or collection of strings; got %r: %r' % (name, type(value), value))
        value = frozenset(value)
        for item in value:
            if not isinstance(item, str):
                raise TypeError('%s: expected `None` or collection of strings; got item of type %r: %r' % (name, type(item), item))
        return value

    @property
    def runs(self):
        return self._runs

    @property
    def tags(self):
        return self._tags

    def __repr__(self):
        return 'RunTagFilter(%s)' % ', '.join(('runs=%r' % (self._runs,), 'tags=%r' % (self._tags,)))