from unittest import mock
import pytest
import cirq
import cirq_google as cg
def _set_get_processor_return(get_processor):
    from google.protobuf.text_format import Merge
    from cirq_google.api import v2
    from cirq_google.engine import util
    from cirq_google.cloud import quantum
    device_spec = util.pack_any(Merge("\nvalid_gate_sets: [{\n    name: 'test_set',\n    valid_gates: [{\n        id: 'x',\n        number_of_qubits: 1,\n        gate_duration_picos: 1000,\n        valid_targets: ['1q_targets']\n    }]\n}],\nvalid_qubits: ['0_0', '1_1'],\nvalid_targets: [{\n    name: '1q_targets',\n    target_ordering: SYMMETRIC,\n    targets: [{\n        ids: ['0_0']\n    }]\n}]\n", v2.device_pb2.DeviceSpecification()))
    get_processor.return_value = quantum.QuantumProcessor(device_spec=device_spec)
    return get_processor