import abc
import dataclasses
from typing import Iterator, Optional, Type, TypeVar
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
Stops tracker from getting updates on model changes in self.

        An error will be raised if tracker is not a StorageUpdateTracker created by
        this Model that has not previously been removed.

        Using an UpdateTracker (via checkpoint or export_update) after it has been
        removed will result in an error.

        Args:
          tracker: The StorageUpdateTracker to unregister.

        Raises:
          KeyError: The tracker was created by another model or was already removed.
        