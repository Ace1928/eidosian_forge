import re
import pytest
import cirq
def test_record_channel_measurement():
    cd = cirq.ClassicalDataDictionaryStore()
    cd.record_channel_measurement(mkey_m, 1)
    assert cd.channel_records == {mkey_m: [1]}
    assert cd.keys() == (mkey_m,)