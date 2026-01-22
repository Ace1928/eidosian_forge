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
def _draw_moment_annotations(*, moment: 'cirq.Moment', col: int, use_unicode_characters: bool, label_map: Dict['cirq.LabelEntity', int], out_diagram: 'cirq.TextDiagramDrawer', precision: Optional[int], get_circuit_diagram_info: Callable[['cirq.Operation', 'cirq.CircuitDiagramInfoArgs'], 'cirq.CircuitDiagramInfo'], include_tags: bool, first_annotation_row: int, transpose: bool):
    for k, annotation in enumerate(_get_moment_annotations(moment)):
        args = protocols.CircuitDiagramInfoArgs(known_qubits=(), known_qubit_count=0, use_unicode_characters=use_unicode_characters, label_map=label_map, precision=precision, include_tags=include_tags, transpose=transpose)
        info = get_circuit_diagram_info(annotation, args)
        symbols = info._wire_symbols_including_formatted_exponent(args)
        text = symbols[0] if symbols else str(annotation)
        out_diagram.force_vertical_padding_after(first_annotation_row + k - 1, 0)
        out_diagram.write(col, first_annotation_row + k, text)