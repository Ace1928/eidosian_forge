import pytest
import cirq
def test_qasm_output_args_format():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    m_a = cirq.measure(a, key='meas_a')
    m_b = cirq.measure(b, key='meas_b')
    args = cirq.QasmArgs(precision=4, version='2.0', qubit_id_map={a: 'aaa[0]', b: 'bbb[0]'}, meas_key_id_map={'meas_a': 'm_a', 'meas_b': 'm_b'})
    assert args.format('_{0}_', a) == '_aaa[0]_'
    assert args.format('_{0}_', b) == '_bbb[0]_'
    assert args.format('_{0:meas}_', cirq.measurement_key_name(m_a)) == '_m_a_'
    assert args.format('_{0:meas}_', cirq.measurement_key_name(m_b)) == '_m_b_'
    assert args.format('_{0}_', 89.1234567) == '_89.1235_'
    assert args.format('_{0}_', 1.23) == '_1.23_'
    assert args.format('_{0:half_turns}_', 89.1234567) == '_pi*89.1235_'
    assert args.format('_{0:half_turns}_', 1.23) == '_pi*1.23_'
    assert args.format('_{0}_', 'other') == '_other_'