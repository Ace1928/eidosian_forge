from typing import Type, Union, List, Optional
from fnmatch import fnmatch
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.converters.circuit_to_dag import circuit_to_dag
from qiskit.circuit.gate import Gate
Call a decomposition pass on this circuit,
        to decompose one level (shallow decompose).