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
def test_run_job_over_stream_processor_unset_raises():
    client = EngineClient()
    with pytest.raises(ValueError, match='Must specify a processor id'):
        client.run_job_over_stream(project_id='proj', program_id='prog', code=any_pb2.Any(), job_id='job0', processor_id='', run_context=any_pb2.Any())