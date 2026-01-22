from unittest import mock
import pytest
import cirq
import cirq_google as cg
class _ExampleProcessorRecord(cg.ProcessorRecord):

    def get_processor(self) -> 'cg.engine.AbstractProcessor':
        return cg.engine.SimulatedLocalProcessor(processor_id='example')