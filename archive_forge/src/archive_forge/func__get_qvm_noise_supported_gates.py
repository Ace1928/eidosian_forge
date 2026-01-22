from pyquil.quilbase import Gate
from pyquil.quilatom import Parameter, unpack_qubit
from pyquil.external.rpcq import CompilerISA, Supported1QGate, Supported2QGate, GateInfo, Edge
from typing import List, Optional
import logging
def _get_qvm_noise_supported_gates(isa: CompilerISA) -> List[Gate]:
    """
    Generate the gate set associated with an ISA for which QVM noise is supported.

    :param isa: The instruction set architecture for a QPU.
    :return: A list of Gate objects encapsulating all gates compatible with the ISA.
    """
    gates = []
    for _qubit_id, q in isa.qubits.items():
        if q.dead:
            continue
        for gate in q.gates:
            if gate.operator == Supported1QGate.MEASURE:
                continue
            assert isinstance(gate, GateInfo)
            qvm_noise_supported_gate = _transform_rpcq_qubit_gate_info_to_qvm_noise_supported_gate(qubit_id=q.id, gate=gate)
            if qvm_noise_supported_gate is not None:
                gates.append(qvm_noise_supported_gate)
    for _edge_id, edge in isa.edges.items():
        if edge.dead:
            continue
        qvm_noise_supported_gates = _transform_rpcq_edge_gate_info_to_qvm_noise_supported_gates(edge)
        gates.extend(qvm_noise_supported_gates)
    return gates