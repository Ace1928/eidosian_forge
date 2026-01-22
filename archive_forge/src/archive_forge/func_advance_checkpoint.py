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
def advance_checkpoint(self) -> None:
    """Track changes to the model only after this function call."""
    return self.storage_update_tracker.advance_checkpoint()