from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional, List
import rpcq
from qcs_api_client.client import QCSClientConfiguration
from rpcq.messages import TargetDevice as TargetQuantumProcessor
class CompilerClient:
    """
    Client for making requests to a Quil compiler.
    """

    def __init__(self, *, client_configuration: QCSClientConfiguration, request_timeout: float=10.0) -> None:
        """
        Instantiate a new compiler client.

        :param client_configuration: Configuration for client.
        :param request_timeout: Timeout for requests, in seconds.
        """
        base_url = client_configuration.profile.applications.pyquil.quilc_url
        if not base_url.startswith('tcp://'):
            raise ValueError(f"Expected compiler URL '{base_url}' to start with 'tcp://'")
        self.base_url = base_url
        self.timeout = request_timeout

    def get_version(self) -> str:
        """
        Get version info for compiler server.
        """
        with self._rpcq_client() as rpcq_client:
            version: Optional[str] = rpcq_client.call('get_version_info').get('quilc')
            if version is None:
                raise ValueError("Expected compiler version info to contain a 'quilc' field.")
            return version

    def compile_to_native_quil(self, request: CompileToNativeQuilRequest) -> CompileToNativeQuilResponse:
        """
        Compile Quil program to native Quil.
        """
        rpcq_request = rpcq.messages.NativeQuilRequest(quil=request.program, target_device=request.target_quantum_processor)
        with self._rpcq_client() as rpcq_client:
            response: rpcq.messages.NativeQuilResponse = rpcq_client.call('quil_to_native_quil', rpcq_request, protoquil=request.protoquil)
            metadata: Optional[NativeQuilMetadataResponse] = None
            if response.metadata is not None:
                metadata = NativeQuilMetadataResponse(final_rewiring=response.metadata.final_rewiring, gate_depth=response.metadata.gate_depth, gate_volume=response.metadata.gate_volume, multiqubit_gate_depth=response.metadata.multiqubit_gate_depth, program_duration=response.metadata.program_duration, program_fidelity=response.metadata.program_fidelity, topological_swaps=response.metadata.topological_swaps, qpu_runtime_estimation=response.metadata.qpu_runtime_estimation)
            return CompileToNativeQuilResponse(native_program=response.quil, metadata=metadata)

    def conjugate_pauli_by_clifford(self, request: ConjugatePauliByCliffordRequest) -> ConjugatePauliByCliffordResponse:
        """
        Conjugate a Pauli element by a Clifford element.
        """
        rpcq_request = rpcq.messages.ConjugateByCliffordRequest(pauli=rpcq.messages.PauliTerm(indices=request.pauli_indices, symbols=request.pauli_symbols), clifford=request.clifford)
        with self._rpcq_client() as rpcq_client:
            response: rpcq.messages.ConjugateByCliffordResponse = rpcq_client.call('conjugate_pauli_by_clifford', rpcq_request)
            return ConjugatePauliByCliffordResponse(phase_factor=response.phase, pauli=response.pauli)

    def generate_randomized_benchmarking_sequence(self, request: GenerateRandomizedBenchmarkingSequenceRequest) -> GenerateRandomizedBenchmarkingSequenceResponse:
        """
        Generate a randomized benchmarking sequence.
        """
        rpcq_request = rpcq.messages.RandomizedBenchmarkingRequest(depth=request.depth, qubits=request.num_qubits, gateset=request.gateset, seed=request.seed, interleaver=request.interleaver)
        with self._rpcq_client() as rpcq_client:
            response: rpcq.messages.RandomizedBenchmarkingResponse = rpcq_client.call('generate_rb_sequence', rpcq_request)
            return GenerateRandomizedBenchmarkingSequenceResponse(sequence=response.sequence)

    @contextmanager
    def _rpcq_client(self) -> Iterator[rpcq.Client]:
        client = rpcq.Client(endpoint=self.base_url, timeout=self.timeout)
        try:
            yield client
        finally:
            client.close()