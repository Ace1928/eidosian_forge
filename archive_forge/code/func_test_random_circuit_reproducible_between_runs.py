from typing import Optional, Dict, Sequence, Union, cast
import random
import numpy as np
import pytest
import cirq
import cirq.testing
def test_random_circuit_reproducible_between_runs():
    circuit = cirq.testing.random_circuit(5, 8, 0.5, random_state=77)
    expected_diagram = '\n                  ┌──┐\n0: ────────────────S─────iSwap───────Y───X───\n                         │\n1: ───────────Y──────────iSwap───────Y───────\n\n2: ─────────────────X────T───────────S───S───\n                    │\n3: ───────@────────S┼────H───────────────Z───\n          │         │\n4: ───────@─────────@────────────────────X───\n                  └──┘\n    '
    cirq.testing.assert_has_diagram(circuit, expected_diagram)