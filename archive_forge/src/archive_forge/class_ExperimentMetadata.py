from typing import Collection, Sequence, Tuple, Union
import abc
import dataclasses
import enum
import numpy as np
class ExperimentMetadata:
    """Metadata about an experiment.

    All fields have default values: i.e., they will always be present on
    the object, but may be omitted in a constructor call.

    Attributes:
      data_location: A human-readable description of the data source, such as a
        path to a directory on disk.
      experiment_name: A user-facing name for the experiment (as a `str`).
      experiment_description: A user-facing description for the experiment
        (as a `str`).
      creation_time: A timestamp for the creation of the experiment, as `float`
        seconds since the epoch.
    """

    def __init__(self, *, data_location='', experiment_name='', experiment_description='', creation_time=0):
        self._data_location = data_location
        self._experiment_name = experiment_name
        self._experiment_description = experiment_description
        self._creation_time = creation_time

    @property
    def data_location(self):
        return self._data_location

    @property
    def experiment_name(self):
        return self._experiment_name

    @property
    def experiment_description(self):
        return self._experiment_description

    @property
    def creation_time(self):
        return self._creation_time

    def _as_tuple(self):
        """Helper for `__eq__` and `__hash__`."""
        return (self._data_location, self._experiment_name, self._experiment_description, self._creation_time)

    def __eq__(self, other):
        if not isinstance(other, ExperimentMetadata):
            return False
        return self._as_tuple() == other._as_tuple()

    def __hash__(self):
        return hash(self._as_tuple())

    def __repr__(self):
        return 'ExperimentMetadata(%s)' % ', '.join(('data_location=%r' % (self.data_location,), 'experiment_name=%r' % (self._experiment_name,), 'experiment_description=%r' % (self._experiment_description,), 'creation_time=%r' % (self._creation_time,)))