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
def test_update_reservation(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)
    name = 'projects/proj/processors/processor0/reservations/papar-party-44'
    result = quantum.QuantumReservation(name=name, start_time=Timestamp(seconds=1000001000), end_time=Timestamp(seconds=1000002000), whitelisted_users=['jeff@google.com'])
    grpc_client.update_quantum_reservation.return_value = result
    client = EngineClient()
    assert client.update_reservation('proj', 'processor0', 'papar-party-44', start=datetime.datetime.fromtimestamp(1000001000), end=datetime.datetime.fromtimestamp(1000002000), whitelisted_users=['jeff@google.com']) == result
    grpc_client.update_quantum_reservation.assert_called_with(quantum.UpdateQuantumReservationRequest(name=name, quantum_reservation=result, update_mask=FieldMask(paths=['start_time', 'end_time', 'whitelisted_users'])))