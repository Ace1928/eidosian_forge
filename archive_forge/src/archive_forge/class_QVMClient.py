import re
from contextlib import contextmanager
from dataclasses import dataclass
from json.decoder import JSONDecodeError
from typing import Iterator, Any, Dict, Union, Tuple, Optional, List, cast
import httpx
from qcs_api_client.client import QCSClientConfiguration
from pyquil.api._errors import ApiError, UnknownApiError, TooManyQubitsError, error_mapping
class QVMClient:
    """
    Client for making requests to a Quantum Virtual Machine.
    """

    def __init__(self, *, client_configuration: QCSClientConfiguration, request_timeout: float=10.0) -> None:
        """
        Instantiate a new compiler client.

        :param client_configuration: Configuration for client.
        :param request_timeout: Timeout for requests, in seconds.
        """
        self.base_url = client_configuration.profile.applications.pyquil.qvm_url
        self.timeout = request_timeout

    def get_version(self) -> str:
        """
        Get version info for QVM server.
        """
        return self._post_json({'type': 'version'}).text.split()[0]

    def run_program(self, request: RunProgramRequest) -> RunProgramResponse:
        """
        Run a Quil program and return its results.
        """
        payload: Dict[str, Any] = {'type': 'multishot', 'compiled-quil': request.program, 'addresses': request.addresses, 'trials': request.trials}
        if request.measurement_noise is not None:
            payload['measurement-noise'] = request.measurement_noise
        if request.gate_noise is not None:
            payload['gate-noise'] = request.gate_noise
        if request.seed is not None:
            payload['rng-seed'] = request.seed
        return RunProgramResponse(results=cast(Dict[str, List[List[int]]], self._post_json(payload).json()))

    def run_and_measure_program(self, request: RunAndMeasureProgramRequest) -> RunAndMeasureProgramResponse:
        """
        Run and measure a Quil program, and return its results.
        """
        payload: Dict[str, Any] = {'type': 'multishot-measure', 'compiled-quil': request.program, 'qubits': request.qubits, 'trials': request.trials}
        if request.measurement_noise is not None:
            payload['measurement-noise'] = request.measurement_noise
        if request.gate_noise is not None:
            payload['gate-noise'] = request.gate_noise
        if request.seed is not None:
            payload['rng-seed'] = request.seed
        return RunAndMeasureProgramResponse(results=cast(List[List[int]], self._post_json(payload).json()))

    def measure_expectation(self, request: MeasureExpectationRequest) -> MeasureExpectationResponse:
        """
        Measure expectation value of Pauli operators given a defined state.
        """
        payload: Dict[str, Any] = {'type': 'expectation', 'state-preparation': request.prep_program, 'operators': request.pauli_operators}
        if request.seed is not None:
            payload['rng-seed'] = request.seed
        return MeasureExpectationResponse(expectations=cast(List[float], self._post_json(payload).json()))

    def get_wavefunction(self, request: GetWavefunctionRequest) -> GetWavefunctionResponse:
        """
        Run a program and retrieve the resulting wavefunction.
        """
        payload: Dict[str, Any] = {'type': 'wavefunction', 'compiled-quil': request.program}
        if request.measurement_noise is not None:
            payload['measurement-noise'] = request.measurement_noise
        if request.gate_noise is not None:
            payload['gate-noise'] = request.gate_noise
        if request.seed is not None:
            payload['rng-seed'] = request.seed
        return GetWavefunctionResponse(wavefunction=self._post_json(payload).content)

    def _post_json(self, json: Dict[str, Any]) -> httpx.Response:
        with self._http_client() as http:
            response = http.post('/', json=json)
            if response.status_code >= 400:
                raise self._parse_error(response)
        return response

    @contextmanager
    def _http_client(self) -> Iterator[httpx.Client]:
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            yield client

    @staticmethod
    def _parse_error(res: httpx.Response) -> ApiError:
        """
        Errors should contain a "status" field with a human readable explanation of
        what went wrong as well as a "error_type" field indicating the kind of error that can be mapped
        to a Python type.

        There's a fallback error UnknownApiError for other types of exceptions (network issues, api
        gateway problems, etc.)
        """
        try:
            body = res.json()
        except JSONDecodeError:
            raise UnknownApiError(res.text)
        if 'error_type' not in body:
            raise UnknownApiError(str(body))
        error_type = body['error_type']
        status = body['status']
        if re.search('[0-9]+ qubits were requested, but the QVM is limited to [0-9]+ qubits.', status):
            return TooManyQubitsError(status)
        error_cls = error_mapping.get(error_type, UnknownApiError)
        return error_cls(status)