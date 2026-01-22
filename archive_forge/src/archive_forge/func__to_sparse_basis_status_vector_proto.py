import dataclasses
import enum
from typing import Dict, Optional, TypeVar
from ortools.math_opt import solution_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import sparse_containers
def _to_sparse_basis_status_vector_proto(terms: Dict[T, BasisStatus]) -> solution_pb2.SparseBasisStatusVector:
    """Converts a basis vector from a python Dict to a protocol buffer."""
    result = solution_pb2.SparseBasisStatusVector()
    if terms:
        id_and_status = sorted(((key.id, status.value) for key, status in terms.items()))
        ids, values = zip(*id_and_status)
        result.ids[:] = ids
        result.values[:] = values
    return result