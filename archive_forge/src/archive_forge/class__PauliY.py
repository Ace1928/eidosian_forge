import abc
from typing import Any, cast, Tuple, TYPE_CHECKING, Union, Dict
from cirq._doc import document
from cirq.ops import common_gates, raw_types, identity
from cirq.type_workarounds import NotImplementedType
class _PauliY(Pauli, common_gates.YPowGate):

    def __init__(self):
        Pauli.__init__(self, index=1, name='Y')
        common_gates.YPowGate.__init__(self, exponent=1.0)

    def __pow__(self, exponent: 'cirq.TParamVal') -> common_gates.YPowGate:
        return common_gates.YPowGate(exponent=exponent) if exponent != 1 else _PauliY()

    def _with_exponent(self, exponent: 'cirq.TParamVal') -> common_gates.YPowGate:
        return self.__pow__(exponent)

    @classmethod
    def _from_json_dict_(cls, exponent, global_shift, **kwargs):
        assert global_shift == 0
        assert exponent == 1
        return Pauli._XYZ[1]

    @property
    def basis(self) -> Dict[int, '_YEigenState']:
        from cirq.value.product_state import _YEigenState
        return {+1: _YEigenState(+1), -1: _YEigenState(-1)}