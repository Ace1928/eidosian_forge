from unittest import mock
import pytest
import cirq
import cirq_google as cg
from cirq_google.engine.abstract_processor import AbstractProcessor
@pytest.mark.parametrize('run_name, device_config_name', [('run_name', ''), ('', 'device_config_name')])
def test_processor_sampler_with_invalid_configuration_throws(run_name, device_config_name):
    processor = mock.create_autospec(AbstractProcessor)
    with pytest.raises(ValueError, match='Cannot specify only one of `run_name` and `device_config_name`'):
        cg.ProcessorSampler(processor=processor, run_name=run_name, device_config_name=device_config_name)