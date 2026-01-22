from typing import Collection, Sequence, Tuple, Union
import abc
import dataclasses
import enum
import numpy as np
class ScalarTimeSeries(_TimeSeries):
    """Metadata about a scalar time series for a particular run and tag.

    Attributes:
      max_step: The largest step value of any datum in this scalar time series; a
        nonnegative integer.
      max_wall_time: The largest wall time of any datum in this time series, as
        `float` seconds since epoch.
      plugin_content: A bytestring of arbitrary plugin-specific metadata for this
        time series, as provided to `tf.summary.write` in the
        `plugin_data.content` field of the `metadata` argument.
      description: An optional long-form Markdown description, as a `str` that is
        empty if no description was specified.
      display_name: An optional long-form Markdown description, as a `str` that is
        empty if no description was specified. Deprecated; may be removed soon.
    """

    def __eq__(self, other):
        if not isinstance(other, ScalarTimeSeries):
            return False
        if self._max_step != other._max_step:
            return False
        if self._max_wall_time != other._max_wall_time:
            return False
        if self._plugin_content != other._plugin_content:
            return False
        if self._description != other._description:
            return False
        if self._display_name != other._display_name:
            return False
        return True

    def __hash__(self):
        return hash((self._max_step, self._max_wall_time, self._plugin_content, self._description, self._display_name))

    def __repr__(self):
        return 'ScalarTimeSeries(%s)' % ', '.join(('max_step=%r' % (self._max_step,), 'max_wall_time=%r' % (self._max_wall_time,), 'plugin_content=%r' % (self._plugin_content,), 'description=%r' % (self._description,), 'display_name=%r' % (self._display_name,)))