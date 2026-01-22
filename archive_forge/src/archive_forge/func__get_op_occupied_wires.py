from pennylane.ops import Conditional
from pennylane.measurements import MidMeasureMP, MeasurementProcess
from .utils import default_wire_map
def _get_op_occupied_wires(op, wire_map, bit_map):
    """Helper function to find wires that would be used by an operator in a drawable layer."""
    if isinstance(op, MidMeasureMP):
        mapped_wire = wire_map[op.wires[0]]
        if op in bit_map:
            min_wire = mapped_wire
            max_wire = max(wire_map.values())
            return set(range(min_wire, max_wire + 1))
        return {mapped_wire}
    if isinstance(op, Conditional):
        mapped_wires = [wire_map[wire] for wire in op.then_op.wires]
        min_wire = min(mapped_wires)
        max_wire = max(wire_map.values())
        return set(range(min_wire, max_wire + 1))
    if len(op.wires) == 0:
        mapped_wires = set(wire_map.values())
        return mapped_wires
    mapped_wires = {wire_map[wire] for wire in op.wires}
    min_wire = min(mapped_wires)
    max_wire = max(mapped_wires)
    return set(range(min_wire, max_wire + 1))