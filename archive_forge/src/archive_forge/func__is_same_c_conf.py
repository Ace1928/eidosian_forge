from qiskit.circuit.controlledgate import ControlledGate
def _is_same_c_conf(self, node_circuit, node_template):
    """
        Check if the clbits configurations are compatible.
        Args:
            node_circuit (DAGDepNode): node in the circuit.
            node_template (DAGDepNode): node in the template.
        Returns:
            bool: True if possible, False otherwise.
        """
    if node_circuit.type == 'op' and getattr(node_circuit.op, 'condition', None) and (node_template.type == 'op') and getattr(node_template.op, 'condition', None):
        if set(self.carg_indices) != set(node_template.cindices):
            return False
        if getattr(node_circuit.op, 'condition', None)[1] != getattr(node_template.op, 'condition', None)[1]:
            return False
    return True