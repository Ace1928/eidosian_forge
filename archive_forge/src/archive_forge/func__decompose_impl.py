import itertools
from typing import Optional
from unittest import mock
import pytest
import cirq
def _decompose_impl(self, qubits, mock_qm: mock.Mock):
    mock_qm.qalloc(self.recurse)
    yield (RecursiveDecompose(recurse=False, mock_qm=self.mock_qm, with_context=self.with_context).on(*qubits) if self.recurse else cirq.Z.on_each(*qubits))
    mock_qm.qfree(self.recurse)