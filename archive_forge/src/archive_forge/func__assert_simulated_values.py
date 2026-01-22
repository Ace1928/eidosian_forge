import sys
import unittest.mock as mock
import pytest
import cirq_google as cg
from cirq_google.engine.qcs_notebook import get_qcs_objects_for_notebook, QCSObjectsForNotebook
def _assert_simulated_values(result: QCSObjectsForNotebook):
    assert not result.signed_in
    assert result.is_simulator
    assert result.project_id == 'fake_project'