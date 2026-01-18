import re
import numpy as np
import pytest
import sympy
import cirq
def test_two_qubit_diagram():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    c = cirq.Circuit(cirq.MatrixGate(cirq.unitary(cirq.CZ)).on(a, b), cirq.MatrixGate(cirq.unitary(cirq.CZ)).on(c, a))
    assert re.match('\n      ┌[          ]+┐\n      │[0-9\\.+\\-j ]+│\na: ───│[0-9\\.+\\-j ]+│───#2─+\n      │[0-9\\.+\\-j ]+│   │\n      │[0-9\\.+\\-j ]+│   │\n      └[          ]+┘   │\n      │[          ]+    │\nb: ───#2[─────────]+────┼──+\n       [          ]+    │\n       [          ]+    ┌[          ]+┐\n       [          ]+    │[0-9\\.+\\-j ]+│\nc: ────[──────────]+────│[0-9\\.+\\-j ]+│──+\n       [          ]+    │[0-9\\.+\\-j ]+│\n       [          ]+    │[0-9\\.+\\-j ]+│\n       [          ]+    └[          ]+┘\n    '.strip(), c.to_text_diagram().strip())
    assert re.match('\na[          ]+  b  c\n│[          ]+  │  │\n┌[          ]+┐ │  │\n│[0-9\\.+\\-j ]+│ │  │\n│[0-9\\.+\\-j ]+│─#2 │\n│[0-9\\.+\\-j ]+│ │  │\n│[0-9\\.+\\-j ]+│ │  │\n└[          ]+┘ │  │\n│[          ]+  │  │\n│[          ]+  │  ┌[          ]+┐\n│[          ]+  │  │[0-9\\.+\\-j ]+│\n#2[─────────]+──┼──│[0-9\\.+\\-j ]+│\n│[          ]+  │  │[0-9\\.+\\-j ]+│\n│[          ]+  │  │[0-9\\.+\\-j ]+│\n│[          ]+  │  └[          ]+┘\n│[          ]+  │  │\n    '.strip(), c.to_text_diagram(transpose=True).strip())