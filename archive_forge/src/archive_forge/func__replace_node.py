import time
import logging
from functools import singledispatchmethod
from itertools import zip_longest
from collections import defaultdict
import rustworkx
from qiskit.circuit import (
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.equivalence import Key, NodeData
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
def _replace_node(self, dag, node, instr_map):
    target_params, target_dag = instr_map[node.op.name, node.op.num_qubits]
    if len(node.op.params) != len(target_params):
        raise TranspilerError('Translation num_params not equal to op num_params.Op: {} {} Translation: {}\n{}'.format(node.op.params, node.op.name, target_params, target_dag))
    if node.op.params:
        parameter_map = dict(zip(target_params, node.op.params))
        bound_target_dag = target_dag.copy_empty_like()
        for inner_node in target_dag.topological_op_nodes():
            if any((isinstance(x, ParameterExpression) for x in inner_node.op.params)):
                new_op = inner_node.op.copy()
                new_params = []
                for param in new_op.params:
                    if not isinstance(param, ParameterExpression):
                        new_params.append(param)
                    else:
                        bind_dict = {x: parameter_map[x] for x in param.parameters}
                        if any((isinstance(x, ParameterExpression) for x in bind_dict.values())):
                            new_value = param
                            for x in bind_dict.items():
                                new_value = new_value.assign(*x)
                        else:
                            new_value = param.bind(bind_dict)
                        if not new_value.parameters:
                            new_value = new_value.numeric()
                        new_params.append(new_value)
                new_op.params = new_params
            else:
                new_op = inner_node.op
            bound_target_dag.apply_operation_back(new_op, inner_node.qargs, inner_node.cargs)
        if isinstance(target_dag.global_phase, ParameterExpression):
            old_phase = target_dag.global_phase
            bind_dict = {x: parameter_map[x] for x in old_phase.parameters}
            if any((isinstance(x, ParameterExpression) for x in bind_dict.values())):
                new_phase = old_phase
                for x in bind_dict.items():
                    new_phase = new_phase.assign(*x)
            else:
                new_phase = old_phase.bind(bind_dict)
            if not new_phase.parameters:
                new_phase = new_phase.numeric()
                if isinstance(new_phase, complex):
                    raise TranspilerError(f"Global phase must be real, but got '{new_phase}'")
            bound_target_dag.global_phase = new_phase
    else:
        bound_target_dag = target_dag
    if len(bound_target_dag.op_nodes()) == 1 and len(bound_target_dag.op_nodes()[0].qargs) == len(node.qargs):
        dag_op = bound_target_dag.op_nodes()[0].op
        if getattr(node.op, 'condition', None):
            dag_op = dag_op.copy()
        dag.substitute_node(node, dag_op, inplace=True)
        if bound_target_dag.global_phase:
            dag.global_phase += bound_target_dag.global_phase
    else:
        dag.substitute_node_with_dag(node, bound_target_dag)