import sys
import unittest.mock as mock
import pytest
import cirq_google as cg
from cirq_google.engine.qcs_notebook import get_qcs_objects_for_notebook, QCSObjectsForNotebook
def _assert_correct_types(result: QCSObjectsForNotebook):
    assert isinstance(result.device, cg.GridDevice)
    assert isinstance(result.sampler, cg.ProcessorSampler)
    assert isinstance(result.engine, cg.engine.AbstractEngine)
    assert isinstance(result.processor, cg.engine.AbstractProcessor)