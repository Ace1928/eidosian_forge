import abc
from typing import Any, cast, Tuple, TYPE_CHECKING, Union, Dict
from cirq._doc import document
from cirq.ops import common_gates, raw_types, identity
from cirq.type_workarounds import NotImplementedType
class _PauliX(Pauli, common_gates.XPowGate):

    def __init__(self):
        Pauli.__init__(self, index=0, name='X')
        common_gates.XPowGate.__init__(self, exponent=1.0)

    def __pow__(self, exponent: 'cirq.TParamVal') -> common_gates.XPowGate:
        return common_gates.XPowGate(exponent=exponent) if exponent != 1 else _PauliX()

    def _with_exponent(self, exponent: 'cirq.TParamVal') -> common_gates.XPowGate:
        return self.__pow__(exponent)

    @classmethod
    def _from_json_dict_(cls, exponent, global_shift, **kwargs):
        assert global_shift == 0
        assert exponent == 1
        return Pauli._XYZ[0]

    @property
    def basis(self) -> Dict[int, '_XEigenState']:
        from cirq.value.product_state import _XEigenState
        return {+1: _XEigenState(+1), -1: _XEigenState(-1)}