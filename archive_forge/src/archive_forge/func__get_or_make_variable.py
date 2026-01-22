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
def _get_or_make_variable(self, variable_id: int) -> Variable:
    result = self._variable_ids.get(variable_id)
    if result:
        return result
    result = Variable(self, variable_id)
    self._variable_ids[variable_id] = result
    return result