import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
class _TestDecomposingChannel(cirq.Gate):

    def __init__(self, channels):
        self.channels = channels

    def _qid_shape_(self):
        return tuple((d for chan in self.channels for d in cirq.qid_shape(chan)))

    def _decompose_(self, qubits):
        return [chan.on(q) for chan, q in zip(self.channels, qubits)]