import pytest
import numpy as np
import google.protobuf.text_format as text_format
import cirq
import cirq_google as cg
import cirq_google.api.v2 as v2
import cirq_google.engine.virtual_engine_factory as factory
@pytest.mark.parametrize('processor_id', ['rainbow', 'weber'])
def test_sample_device_zphase(processor_id):
    zphase_data = factory.load_sample_device_zphase(processor_id)
    assert 'sqrt_iswap' in zphase_data
    sqrt_iswap_data = zphase_data['sqrt_iswap']
    for angle in ['zeta', 'gamma']:
        assert angle in sqrt_iswap_data
        for (q0, q1), val in sqrt_iswap_data[angle].items():
            assert isinstance(q0, cirq.Qid)
            assert isinstance(q1, cirq.Qid)
            assert isinstance(val, float)