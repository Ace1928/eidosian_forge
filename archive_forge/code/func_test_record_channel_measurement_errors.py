import re
import pytest
import cirq
def test_record_channel_measurement_errors():
    cd = cirq.ClassicalDataDictionaryStore()
    cd.record_channel_measurement(mkey_m, 1)
    cd.record_channel_measurement(mkey_m, 1)
    with pytest.raises(ValueError, match='Channel Measurement already logged to key m'):
        cd.record_measurement(mkey_m, (0, 1), two_qubits)
    cd = cirq.ClassicalDataDictionaryStore()
    cd.record_measurement(mkey_m, (0, 1), two_qubits)
    cd.record_measurement(mkey_m, (0, 1), two_qubits)
    with pytest.raises(ValueError, match='Measurement already logged to key m'):
        cd.record_channel_measurement(mkey_m, 1)