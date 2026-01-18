import itertools
from typing import (
from typing_extensions import Self
import numpy as np
from cirq import protocols, ops, qis, _compat
from cirq._import import LazyLoader
from cirq.ops import raw_types, op_tree
from cirq.protocols import circuit_diagram_info_protocol
from cirq.type_workarounds import NotImplementedType
def to_text_diagram(self: 'cirq.Moment', *, xy_breakdown_func: Callable[['cirq.Qid'], Tuple[Any, Any]]=_default_breakdown, extra_qubits: Iterable['cirq.Qid']=(), use_unicode_characters: bool=True, precision: Optional[int]=None, include_tags: bool=True) -> str:
    """Create a text diagram for the moment.

        Args:
            xy_breakdown_func: A function to split qubits/qudits into x and y
                components. For example, the default breakdown turns
                `cirq.GridQubit(row, col)` into the tuple `(col, row)` and
                `cirq.LineQubit(x)` into `(x, 0)`.
            extra_qubits: Extra qubits/qudits to include in the diagram, even
                if they don't have any operations applied in the moment.
            use_unicode_characters: Whether or not the output should use fancy
                unicode characters or stick to plain ASCII. Unicode characters
                look nicer, but some environments don't draw them with the same
                width as ascii characters (which ruins the diagrams).
            precision: How precise numbers, such as angles, should be. Use None
                for infinite precision, or an integer for a certain number of
                digits of precision.
            include_tags: Whether or not to include operation tags in the
                diagram.

        Returns:
            The text diagram rendered into text.
        """
    qs = set(self.qubits) | set(extra_qubits)
    points = {xy_breakdown_func(q) for q in qs}
    x_keys = sorted({pt[0] for pt in points}, key=_SortByValFallbackToType)
    y_keys = sorted({pt[1] for pt in points}, key=_SortByValFallbackToType)
    x_map = {x_key: x + 2 for x, x_key in enumerate(x_keys)}
    y_map = {y_key: y + 2 for y, y_key in enumerate(y_keys)}
    qubit_positions = {}
    for q in qs:
        a, b = xy_breakdown_func(q)
        qubit_positions[q] = (x_map[a], y_map[b])
    diagram = text_diagram_drawer.TextDiagramDrawer()

    def cleanup_key(key: Any) -> Any:
        if isinstance(key, float) and key == int(key):
            return str(int(key))
        return str(key)
    for key, x in x_map.items():
        diagram.write(x, 0, cleanup_key(key))
    for key, y in y_map.items():
        diagram.write(0, y, cleanup_key(key))
    diagram.horizontal_line(1, 0, len(x_map) + 2)
    diagram.vertical_line(1, 0, len(y_map) + 2)
    diagram.force_vertical_padding_after(0, 0)
    diagram.force_vertical_padding_after(1, 0)
    for op in self._sorted_operations_():
        args = protocols.CircuitDiagramInfoArgs(known_qubits=op.qubits, known_qubit_count=len(op.qubits), use_unicode_characters=use_unicode_characters, label_map=None, precision=precision, include_tags=include_tags)
        info = circuit_diagram_info_protocol._op_info_with_fallback(op, args=args)
        symbols = info._wire_symbols_including_formatted_exponent(args)
        for label, q in zip(symbols, op.qubits):
            x, y = qubit_positions[q]
            diagram.write(x, y, label)
        if info.connected:
            for q1, q2 in zip(op.qubits, op.qubits[1:]):
                q1, q2 = sorted([q1, q2])
                x1, y1 = qubit_positions[q1]
                x2, y2 = qubit_positions[q2]
                if x1 != x2:
                    diagram.horizontal_line(y1, x1, x2)
                if y1 != y2:
                    diagram.vertical_line(x2, y1, y2)
    return diagram.render()