import abc
import collections
import dataclasses
import math
import typing
from typing import (
import weakref
import immutabledict
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt.python import hash_model_storage
from ortools.math_opt.python import model_storage
def add_update_tracker(self) -> UpdateTracker:
    """Creates an UpdateTracker registered on this model to view changes."""
    return UpdateTracker(self.storage.add_update_tracker())