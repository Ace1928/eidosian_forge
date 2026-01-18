from unittest import mock
import pytest
import cirq
import cirq_google as cg
from cirq_google.engine.abstract_processor import AbstractProcessor
def test_processor_sampler_processor_property():
    processor = mock.create_autospec(AbstractProcessor)
    sampler = cg.ProcessorSampler(processor=processor)
    assert sampler.processor is processor