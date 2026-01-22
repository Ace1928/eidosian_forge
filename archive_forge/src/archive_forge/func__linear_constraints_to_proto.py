from typing import Dict, Iterable, Iterator, Optional, Set, Tuple
import weakref
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt import sparse_containers_pb2
from ortools.math_opt.python import model_storage
def _linear_constraints_to_proto(linear_constraints: Iterable[Tuple[int, _LinearConstraintStorage]], proto: model_pb2.LinearConstraintsProto) -> None:
    """Exports variables to proto."""
    has_named_lin_con = False
    for _, lin_con_storage in linear_constraints:
        if lin_con_storage.name:
            has_named_lin_con = True
            break
    for lin_con_id, lin_con_storage in linear_constraints:
        proto.ids.append(lin_con_id)
        proto.lower_bounds.append(lin_con_storage.lower_bound)
        proto.upper_bounds.append(lin_con_storage.upper_bound)
        if has_named_lin_con:
            proto.names.append(lin_con_storage.name)