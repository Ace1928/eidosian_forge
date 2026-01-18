import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.dagcircuit import DAGDependency
from qiskit.converters.circuit_to_dagdependency import circuit_to_dagdependency
from qiskit.converters.dagdependency_to_circuit import dagdependency_to_circuit
from qiskit.converters.dag_to_dagdependency import dag_to_dagdependency
from qiskit.converters.dagdependency_to_dag import dagdependency_to_dag
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.library.templates import template_nct_2a_1, template_nct_2a_2, template_nct_2a_3
from qiskit.quantum_info.operators.operator import Operator
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.optimization.template_matching import (

        Args:
            dag(DAGCircuit): DAG circuit.
        Returns:
            DAGCircuit: optimized DAG circuit.
        Raises:
            TranspilerError: If the template has not the right form or
             if the output circuit acts differently as the input circuit.
        