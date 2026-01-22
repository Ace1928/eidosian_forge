from qiskit.transpiler.basepasses import TransformationPass
from qiskit.converters import dag_to_dagdependency, dagdependency_to_dag
from qiskit.dagcircuit.collect_blocks import BlockCollector, BlockCollapser
from qiskit.transpiler.passes.utils import control_flow
Run the CollectLinearFunctions pass on `dag`.
        Args:
            dag (DAGCircuit): the DAG to be optimized.
        Returns:
            DAGCircuit: the optimized DAG.
        