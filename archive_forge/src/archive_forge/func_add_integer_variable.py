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
def add_integer_variable(self, *, lb: float=-math.inf, ub: float=math.inf, name: str='') -> Variable:
    return self.add_variable(lb=lb, ub=ub, is_integer=True, name=name)