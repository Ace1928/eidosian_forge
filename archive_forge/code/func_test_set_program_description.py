import asyncio
import datetime
import os
from unittest import mock
import duet
import pytest
from google.api_core import exceptions
from google.protobuf import any_pb2
from google.protobuf.field_mask_pb2 import FieldMask
from google.protobuf.timestamp_pb2 import Timestamp
from cirq_google.engine.engine_client import EngineClient, EngineException
import cirq_google.engine.stream_manager as engine_stream_manager
from cirq_google.cloud import quantum
@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_set_program_description(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)
    result = quantum.QuantumProgram(name='projects/proj/programs/prog')
    grpc_client.update_quantum_program.return_value = result
    client = EngineClient()
    assert client.set_program_description('proj', 'prog', 'A program') == result
    grpc_client.update_quantum_program.assert_called_with(quantum.UpdateQuantumProgramRequest(name='projects/proj/programs/prog', quantum_program=quantum.QuantumProgram(name='projects/proj/programs/prog', description='A program'), update_mask=FieldMask(paths=['description'])))
    assert client.set_program_description('proj', 'prog', '') == result
    grpc_client.update_quantum_program.assert_called_with(quantum.UpdateQuantumProgramRequest(name='projects/proj/programs/prog', quantum_program=quantum.QuantumProgram(name='projects/proj/programs/prog'), update_mask=FieldMask(paths=['description'])))