import cirq
class GateAllocAndBorrowInDecompose(cirq.Gate):

    def __init__(self, num_alloc: int=1):
        self.num_alloc = num_alloc

    def _num_qubits_(self) -> int:
        return 1

    def __str__(self) -> str:
        return 'TestGate'

    def _decompose_with_context_(self, qubits, context):
        assert context is not None
        qm = context.qubit_manager
        qa, qb = (qm.qalloc(self.num_alloc), qm.qborrow(self.num_alloc))
        for q, b in zip(qa, qb):
            yield cirq.CSWAP(qubits[0], q, b)
        yield cirq.qft(*qb).controlled_by(qubits[0])
        for q, b in zip(qa, qb):
            yield cirq.CSWAP(qubits[0], q, b)
        qm.qfree(qa + qb)