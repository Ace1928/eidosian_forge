import itertools
import uuid
from typing import Iterable
from qiskit.circuit import (
from qiskit.circuit.classical import expr
class DAGOpNode(DAGNode):
    """Object to represent an Instruction at a node in the DAGCircuit."""
    __slots__ = ['op', 'qargs', 'cargs', 'sort_key']

    def __init__(self, op, qargs: Iterable[Qubit]=(), cargs: Iterable[Clbit]=(), dag=None):
        """Create an Instruction node"""
        super().__init__()
        self.op = op
        self.qargs = tuple(qargs)
        self.cargs = tuple(cargs)
        if dag is not None:
            cache_key = (self.qargs, self.cargs)
            key = dag._key_cache.get(cache_key, None)
            if key is not None:
                self.sort_key = key
            else:
                self.sort_key = ','.join((f'{dag.find_bit(q).index:04d}' for q in itertools.chain(*cache_key)))
                dag._key_cache[cache_key] = self.sort_key
        else:
            self.sort_key = str(self.qargs)

    @property
    def name(self):
        """Returns the Instruction name corresponding to the op for this node"""
        return self.op.name

    @name.setter
    def name(self, new_name):
        """Sets the Instruction name corresponding to the op for this node"""
        self.op.name = new_name

    def __repr__(self):
        """Returns a representation of the DAGOpNode"""
        return f'DAGOpNode(op={self.op}, qargs={self.qargs}, cargs={self.cargs})'