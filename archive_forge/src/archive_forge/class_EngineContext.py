import datetime
import enum
import random
import string
from typing import Dict, List, Optional, Sequence, Set, TypeVar, Union, TYPE_CHECKING
import duet
import google.auth
from google.protobuf import any_pb2
import cirq
from cirq._compat import deprecated
from cirq_google.api import v2
from cirq_google.engine import (
from cirq_google.cloud import quantum
from cirq_google.engine.result_type import ResultType
from cirq_google.serialization import CIRCUIT_SERIALIZER, Serializer
from cirq_google.serialization.arg_func_langs import arg_to_proto
@cirq.value_equality
class EngineContext:
    """Context for running against the Quantum Engine API. Most users should
    simply create an Engine object instead of working with one of these
    directly."""

    def __init__(self, proto_version: Optional[ProtoVersion]=None, service_args: Optional[Dict]=None, verbose: Optional[bool]=None, client: 'Optional[engine_client.EngineClient]'=None, timeout: Optional[int]=None, serializer: Serializer=CIRCUIT_SERIALIZER, enable_streaming: bool=False) -> None:
        """Context and client for using Quantum Engine.

        Args:
            proto_version: The version of cirq protos to use. If None, then
                ProtoVersion.V2 will be used.
            service_args: A dictionary of arguments that can be used to
                configure options on the underlying client.
            verbose: Suppresses stderr messages when set to False. Default is
                true.
            client: The engine client to use, if not supplied one will be
                created.
            timeout: Timeout for polling for results, in seconds.  Default is
                to never timeout.
            serializer: Used to serialize circuits when running jobs.
            enable_streaming: Feature gate for making Quantum Engine requests using the stream RPC.
                If True, the Quantum Engine streaming RPC is used for creating jobs
                and getting results. Otherwise, unary RPCs are used.

        Raises:
            ValueError: If either `service_args` and `verbose` were supplied
                or `client` was supplied, or if proto version 1 is specified.
        """
        if (service_args or verbose) and client:
            raise ValueError('either specify service_args and verbose or client')
        self.proto_version = proto_version or ProtoVersion.V2
        if self.proto_version == ProtoVersion.V1:
            raise ValueError('ProtoVersion V1 no longer supported')
        self.serializer = serializer
        self.enable_streaming = enable_streaming
        if not client:
            client = engine_client.EngineClient(service_args=service_args, verbose=verbose)
        self.client = client
        self.timeout = timeout

    def copy(self) -> 'EngineContext':
        return EngineContext(proto_version=self.proto_version, client=self.client)

    def _value_equality_values_(self):
        return (self.proto_version, self.client)

    def _serialize_program(self, program: cirq.AbstractCircuit) -> any_pb2.Any:
        if not isinstance(program, cirq.AbstractCircuit):
            raise TypeError(f'Unrecognized program type: {type(program)}')
        if self.proto_version != ProtoVersion.V2:
            raise ValueError(f'invalid program proto version: {self.proto_version}')
        return util.pack_any(self.serializer.serialize(program))

    def _serialize_run_context(self, sweeps: 'cirq.Sweepable', repetitions: int) -> any_pb2.Any:
        if self.proto_version != ProtoVersion.V2:
            raise ValueError(f'invalid run context proto version: {self.proto_version}')
        return util.pack_any(v2.run_context_to_proto(sweeps, repetitions))