import abc
import enum
import html
import itertools
import math
from collections import defaultdict
from typing import (
from typing_extensions import Self
import networkx
import numpy as np
import cirq._version
from cirq import _compat, devices, ops, protocols, qis
from cirq._doc import document
from cirq.circuits._bucket_priority_queue import BucketPriorityQueue
from cirq.circuits.circuit_operation import CircuitOperation
from cirq.circuits.insert_strategy import InsertStrategy
from cirq.circuits.qasm_output import QasmOutput
from cirq.circuits.text_diagram_drawer import TextDiagramDrawer
from cirq.circuits.moment import Moment
from cirq.protocols import circuit_diagram_info_protocol
from cirq.type_workarounds import NotImplementedType
def _draw_moment_in_diagram(*, moment: 'cirq.Moment', use_unicode_characters: bool, label_map: Dict['cirq.LabelEntity', int], out_diagram: 'cirq.TextDiagramDrawer', precision: Optional[int], moment_groups: List[Tuple[int, int]], get_circuit_diagram_info: Optional[Callable[['cirq.Operation', 'cirq.CircuitDiagramInfoArgs'], 'cirq.CircuitDiagramInfo']], include_tags: bool, first_annotation_row: int, transpose: bool):
    if get_circuit_diagram_info is None:
        get_circuit_diagram_info = circuit_diagram_info_protocol._op_info_with_fallback
    x0 = out_diagram.width()
    non_global_ops = [op for op in moment.operations if op.qubits]
    max_x = x0
    for op in non_global_ops:
        qubits = tuple(op.qubits)
        cbits = tuple(protocols.measurement_keys_touched(op) & label_map.keys())
        labels = qubits + cbits
        indices = [label_map[label] for label in labels]
        y1 = min(indices)
        y2 = max(indices)
        x = x0
        while any((out_diagram.content_present(x, y) for y in range(y1, y2 + 1))):
            out_diagram.force_horizontal_padding_after(x, 0)
            x += 1
        args = protocols.CircuitDiagramInfoArgs(known_qubits=op.qubits, known_qubit_count=len(op.qubits), use_unicode_characters=use_unicode_characters, label_map=label_map, precision=precision, include_tags=include_tags, transpose=transpose)
        info = get_circuit_diagram_info(op, args)
        if y2 > y1 and info.connected:
            out_diagram.vertical_line(x, y1, y2, doubled=len(cbits) != 0)
        symbols = info._wire_symbols_including_formatted_exponent(args, preferred_exponent_index=max(range(len(labels)), key=lambda i: label_map[labels[i]]))
        for s, q in zip(symbols, labels):
            out_diagram.write(x, label_map[q], s)
        if x > max_x:
            max_x = x
    _draw_moment_annotations(moment=moment, use_unicode_characters=use_unicode_characters, col=x0, label_map=label_map, out_diagram=out_diagram, precision=precision, get_circuit_diagram_info=get_circuit_diagram_info, include_tags=include_tags, first_annotation_row=first_annotation_row, transpose=transpose)
    global_phase, tags = _get_global_phase_and_tags_for_ops(moment)
    if global_phase and (global_phase != 1 or not non_global_ops):
        desc = _formatted_phase(global_phase, use_unicode_characters, precision)
        if desc:
            y = max(label_map.values(), default=0) + 1
            if tags and include_tags:
                desc = desc + str(tags)
            out_diagram.write(x0, y, desc)
    if not non_global_ops:
        out_diagram.write(x0, 0, '')
    if moment.operations and max_x > x0:
        moment_groups.append((x0, max_x))