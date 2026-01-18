import cirq
from cirq.ops.named_qubit import _pad_digits
def test_pad_digits():
    assert _pad_digits('') == ''
    assert _pad_digits('a') == 'a'
    assert _pad_digits('a0') == 'a00000000:1'
    assert _pad_digits('a00') == 'a00000000:2'
    assert _pad_digits('a1bc23') == 'a00000001:1bc00000023:2'
    assert _pad_digits('a9') == 'a00000009:1'
    assert _pad_digits('a09') == 'a00000009:2'
    assert _pad_digits('a00000000:8') == 'a00000000:8:00000008:1'