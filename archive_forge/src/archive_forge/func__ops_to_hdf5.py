import json
import typing
from functools import lru_cache
from typing import Dict, FrozenSet, Generic, List, Type, TypeVar
import numpy as np
import pennylane as qml
from pennylane.data.base.attribute import DatasetAttribute
from pennylane.data.base.hdf5 import HDF5Group, h5py
from pennylane.operation import Operator, Tensor
from ._wires import wires_to_json
def _ops_to_hdf5(self, bind_parent: HDF5Group, key: str, value: typing.Sequence[Operator]) -> HDF5Group:
    """Serialize op sequence ``value``, and create nested sequences for any
        composite ops in ``value``.

        Since operators are commonly used in larger composite operations, we handle
        sequences of operators as the default case. This allows for performant (in
        time and space) serialization of large and nested operator sums, products, etc.
        """
    bind = bind_parent.create_group(key)
    op_wire_labels = []
    op_class_names = []
    for i, op in enumerate(value):
        op_key = f'op_{i}'
        if type(op) not in self.consumes_types():
            raise TypeError(f"Serialization of operator type '{type(op).__name__}' is not supported.")
        if isinstance(op, Tensor):
            self._ops_to_hdf5(bind, op_key, op.obs)
            op_wire_labels.append('null')
        elif isinstance(op, qml.Hamiltonian):
            coeffs, ops = op.terms()
            ham_grp = self._ops_to_hdf5(bind, op_key, ops)
            ham_grp['hamiltonian_coeffs'] = coeffs
            op_wire_labels.append('null')
        else:
            bind[op_key] = op.data if len(op.data) else h5py.Empty('f')
            op_wire_labels.append(wires_to_json(op.wires))
        op_class_names.append(type(op).__name__)
    bind['op_wire_labels'] = op_wire_labels
    bind['op_class_names'] = op_class_names
    return bind