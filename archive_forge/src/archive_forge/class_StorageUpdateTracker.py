import abc
import dataclasses
from typing import Iterator, Optional, Type, TypeVar
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
class StorageUpdateTracker(abc.ABC):
    """Tracks updates to an optimization model from a ModelStorage.

    Do not instantiate directly, instead create through
    ModelStorage.add_update_tracker().

    Interacting with an update tracker after it has been removed from the model
    will result in an UsedUpdateTrackerAfterRemovalError error.

    Example:
      mod = model_storage.ModelStorage()
      x = mod.add_variable(0.0, 1.0, True, 'x')
      y = mod.add_variable(0.0, 1.0, True, 'y')
      tracker = mod.add_update_tracker()
      mod.set_variable_ub(x, 3.0)
      tracker.export_update()
        => "variable_updates: {upper_bounds: {ids: [0], values[3.0] }"
      mod.set_variable_ub(y, 2.0)
      tracker.export_update()
        => "variable_updates: {upper_bounds: {ids: [0, 1], values[3.0, 2.0] }"
      tracker.advance_checkpoint()
      tracker.export_update()
        => ""
      mod.set_variable_ub(y, 4.0)
      tracker.export_update()
        => "variable_updates: {upper_bounds: {ids: [1], values[4.0] }"
      tracker.advance_checkpoint()
      mod.remove_update_tracker(tracker)
        => ""
    """

    @abc.abstractmethod
    def export_update(self) -> Optional[model_update_pb2.ModelUpdateProto]:
        """Returns changes to the model since last call to checkpoint/creation, or None if no changes occurred."""
        pass

    @abc.abstractmethod
    def advance_checkpoint(self) -> None:
        """Track changes to the model only after this function call."""
        pass