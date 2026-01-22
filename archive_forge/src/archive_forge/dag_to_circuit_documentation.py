import copy
from qiskit.circuit import QuantumCircuit, CircuitInstruction
Build a ``QuantumCircuit`` object from a ``DAGCircuit``.

    Args:
        dag (DAGCircuit): the input dag.
        copy_operations (bool): Deep copy the operation objects
            in the :class:`~.DAGCircuit` for the output :class:`~.QuantumCircuit`.
            This should only be set to ``False`` if the input :class:`~.DAGCircuit`
            will not be used anymore as the operations in the output
            :class:`~.QuantumCircuit` will be shared instances and
            modifications to operations in the :class:`~.DAGCircuit` will
            be reflected in the :class:`~.QuantumCircuit` (and vice versa).

    Return:
        QuantumCircuit: the circuit representing the input dag.

    Example:
        .. plot::
           :include-source:

           from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
           from qiskit.dagcircuit import DAGCircuit
           from qiskit.converters import circuit_to_dag
           from qiskit.circuit.library.standard_gates import CHGate, U2Gate, CXGate
           from qiskit.converters import dag_to_circuit

           q = QuantumRegister(3, 'q')
           c = ClassicalRegister(3, 'c')
           circ = QuantumCircuit(q, c)
           circ.h(q[0])
           circ.cx(q[0], q[1])
           circ.measure(q[0], c[0])
           circ.rz(0.5, q[1]).c_if(c, 2)
           dag = circuit_to_dag(circ)
           circuit = dag_to_circuit(dag)
           circuit.draw('mpl')
    