import collections
import copyreg
import io
import pickle
import sys
import threading
import traceback
from enum import Enum
import torch
import torch.distributed as dist
from torch._C._distributed_rpc import _get_current_rpc_agent
class _InternalRPCPickler:
    """
    This class provides serialize() and deserialize() interfaces to serialize
    data to be "binary string + tensor table" format
    So for RPC python UDF function and args, non tensor data will be serialized
    into regular binary string, tensor data will be put into thread local tensor
    tables, this serialization format is consistent with builtin operator and args
    using JIT pickler. This format will make tensor handling in C++ much easier,
    e.g. attach tensor to distributed autograd graph in C++
    """

    def __init__(self):
        self._dispatch_table = copyreg.dispatch_table.copy()
        self._dispatch_table[torch.Tensor] = self._tensor_reducer
        self._class_reducer_dict = {}

    def _register_reducer(self, obj_class, reducer):
        if obj_class not in self._class_reducer_dict:
            self._class_reducer_dict[obj_class] = reducer

    @classmethod
    def _tensor_receiver(cls, tensor_index):
        global _thread_local_tensor_tables
        return _thread_local_tensor_tables.recv_tables[tensor_index]

    def _tensor_reducer(self, tensor):
        global _thread_local_tensor_tables
        _thread_local_tensor_tables.send_tables.append(tensor)
        tensor_index = len(_thread_local_tensor_tables.send_tables) - 1
        return (_InternalRPCPickler._tensor_receiver, (tensor_index,))

    @classmethod
    def _py_rref_receiver(cls, rref_fork_data):
        return dist.rpc.PyRRef._deserialize(rref_fork_data)

    def _py_rref_reducer(self, py_rref):
        rref_fork_data = py_rref._serialize()
        return (_InternalRPCPickler._py_rref_receiver, (rref_fork_data,))

    def _rref_reducer(self, rref):
        return self._py_rref_reducer(rref)

    @classmethod
    def _script_module_receiver(cls, script_module_serialized):
        """
        Given a serialized representation of a ScriptModule created with torch.jit.save,
        loads and returns the ScriptModule.
        """
        f = io.BytesIO(script_module_serialized)
        m = torch.jit.load(f)
        return m

    def _script_module_reducer(self, script_module):
        """
        Serializes a ScriptModule.
        """
        f = io.BytesIO()
        torch.jit.save(script_module, f)
        return (_InternalRPCPickler._script_module_receiver, (f.getvalue(),))

    def serialize(self, obj):
        """
        Serialize non tensor data into binary string, tensor data into
        tensor table
        """
        f = io.BytesIO()
        p = _pickler(f)
        p.dispatch_table = self._dispatch_table
        p.dispatch_table[dist.rpc.PyRRef] = self._py_rref_reducer
        p.dispatch_table[dist.rpc.RRef] = self._rref_reducer
        if isinstance(obj, torch.jit.ScriptModule):
            p.dispatch_table[obj.__class__] = self._script_module_reducer
        for class_name in self._class_reducer_dict.keys():
            p.dispatch_table[class_name] = self._class_reducer_dict[class_name]
        global _thread_local_tensor_tables
        if hasattr(_thread_local_tensor_tables, 'send_tables'):
            old_send_tables = _thread_local_tensor_tables.send_tables
        else:
            old_send_tables = None
        _thread_local_tensor_tables.send_tables = []
        p.dump(obj)
        tensors = _thread_local_tensor_tables.send_tables
        if old_send_tables is not None:
            _thread_local_tensor_tables.send_tables = old_send_tables
        else:
            del _thread_local_tensor_tables.send_tables
        return (f.getvalue(), tensors)

    def deserialize(self, binary_data, tensor_table):
        """
        Deserialize binary string + tensor table to original obj
        """
        global _thread_local_tensor_tables
        if hasattr(_thread_local_tensor_tables, 'recv_tables'):
            old_recv_tables = _thread_local_tensor_tables.recv_tables
        else:
            old_recv_tables = None
        _thread_local_tensor_tables.recv_tables = tensor_table
        try:
            unpickler = _unpickler(io.BytesIO(binary_data))
            ret = unpickler.load()
        except AttributeError as e:
            except_str = str(e) + ' Default RPC pickler does not serialize\n            function code. Ensure that UDFs are defined on both caller and\n            callee modules.'
            ret = AttributeError(except_str)
            ret.__cause__ = e
        if old_recv_tables is not None:
            _thread_local_tensor_tables.recv_tables = old_recv_tables
        else:
            del _thread_local_tensor_tables.recv_tables
        return ret