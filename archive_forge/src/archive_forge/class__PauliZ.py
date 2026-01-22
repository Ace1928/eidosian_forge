import abc
from typing import Any, cast, Tuple, TYPE_CHECKING, Union, Dict
from cirq._doc import document
from cirq.ops import common_gates, raw_types, identity
from cirq.type_workarounds import NotImplementedType
class _PauliZ(Pauli, common_gates.ZPowGate):

    def __init__(self):
        Pauli.__init__(self, index=2, name='Z')
        common_gates.ZPowGate.__init__(self, exponent=1.0)

    def __pow__(self, exponent: 'cirq.TParamVal') -> common_gates.ZPowGate:
        return common_gates.ZPowGate(exponent=exponent) if exponent != 1 else _PauliZ()

    def _with_exponent(self, exponent: 'cirq.TParamVal') -> common_gates.ZPowGate:
        return self.__pow__(exponent)

    @classmethod
    def _from_json_dict_(cls, exponent, global_shift, **kwargs):
        assert global_shift == 0
        assert exponent == 1
        return Pauli._XYZ[2]

    @property
    def basis(self) -> Dict[int, '_ZEigenState']:
        from cirq.value.product_state import _ZEigenState
        return {+1: _ZEigenState(+1), -1: _ZEigenState(-1)}