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
def _hdf5_to_ops(self, bind: HDF5Group) -> List[Operator]:
    """Load list of serialized ops from ``bind``."""
    ops = []
    names_bind = bind['op_class_names']
    wires_bind = bind['op_wire_labels']
    op_class_names = [] if names_bind.shape == (0,) else names_bind.asstr()
    op_wire_labels = [] if wires_bind.shape == (0,) else wires_bind.asstr()
    with qml.QueuingManager.stop_recording():
        for i, op_class_name in enumerate(op_class_names):
            op_key = f'op_{i}'
            op_cls = self._supported_ops_dict()[op_class_name]
            if op_cls is Tensor:
                ops.append(Tensor(*self._hdf5_to_ops(bind[op_key])))
            elif op_cls is qml.Hamiltonian:
                ops.append(qml.Hamiltonian(coeffs=list(bind[op_key]['hamiltonian_coeffs']), observables=self._hdf5_to_ops(bind[op_key])))
            else:
                wire_labels = json.loads(op_wire_labels[i])
                op_data = bind[op_key]
                if op_data.shape is not None:
                    params = np.zeros(shape=op_data.shape, dtype=op_data.dtype)
                    op_data.read_direct(params)
                    ops.append(op_cls(*params, wires=wire_labels))
                else:
                    ops.append(op_cls(wires=wire_labels))
    return ops