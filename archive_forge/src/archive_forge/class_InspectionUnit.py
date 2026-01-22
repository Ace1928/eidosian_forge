import dataclasses
import itertools
import os
from typing import Any, Generator, Mapping
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.compat import tf
from tensorboard.compat.proto import event_pb2
@dataclasses.dataclass(frozen=True)
class InspectionUnit:
    """Created for each organizational structure in the event files.

    An InspectionUnit is visible in the final terminal output. For instance, one
    InspectionUnit is created for each subdirectory in logdir. When asked to inspect
    a single event file, there may only be one InspectionUnit.

    Attributes:
      name: Name of the organizational unit that will be printed to console.
      generator: A generator that yields `Event` protos.
      field_to_obs: A mapping from string fields to `Observations` that the inspector
        creates.
    """
    name: str
    generator: Generator[event_pb2.Event, Any, Any]
    field_to_obs: Mapping[str, Observation]