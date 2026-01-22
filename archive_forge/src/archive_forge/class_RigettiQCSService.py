from typing import cast, Optional
import cirq
import httpx
from pyquil import get_qc
from qcs_api_client.operations.sync import (
from qcs_api_client.models import (
from pyquil.api import QuantumComputer
from cirq_rigetti.sampler import RigettiQCSSampler
from cirq_rigetti._qcs_api_client_decorator import _provide_default_client
from cirq_rigetti import circuit_transformers as transformers
from cirq_rigetti import circuit_sweep_executors as executors
class RigettiQCSService:
    """This class supports running circuits on QCS quantum hardware as well as
    pyQuil's quantum virtual machine (QVM). When sampling a parametric circuit
    across a parameter sweep, use `RigettiQCSSampler` instead.

    This class also includes a number of convenience static methods for describing
    Rigetti quantum processors through the `qcs_api_client`."""

    def __init__(self, quantum_computer: QuantumComputer, executor: executors.CircuitSweepExecutor=_default_executor, transformer: transformers.CircuitTransformer=transformers.default):
        """Initializes a `RigettiQCSService`.

        Args:
            quantum_computer: A pyquil.api.QuantumComputer against which to run the `cirq.Circuit`s.
            executor: A callable that first uses the below transformer on cirq.Circuit s and
                then executes the transformed circuit on the quantum_computer. You may pass your
                own callable or any static method on CircuitSweepExecutors.
            transformer: A callable that transforms the cirq.Circuit into a pyquil.Program.
                You may pass your own callable or any static method on CircuitTransformers.
        """
        self._quantum_computer = quantum_computer
        self._executor = executor
        self._transformer = transformer

    def run(self, circuit: cirq.Circuit, repetitions: int, param_resolver: cirq.ParamResolverOrSimilarType=cirq.ParamResolver({})) -> cirq.Result:
        """Run the given circuit on the QuantumComputer with which the user initialized the service.

        Args:
            circuit: The circuit to run.
            repetitions: The number of times to run the circuit.
            param_resolver: A cirq.ParamResolver to resolve parameters in  circuit.

        Returns:
            A cirq.Result.
        """
        results = self._executor(quantum_computer=self._quantum_computer, circuit=circuit, resolvers=[param_resolver], repetitions=repetitions, transformer=self._transformer)
        assert len(results) == 1
        return results[0]

    def sampler(self) -> RigettiQCSSampler:
        """Initializes a cirq.Sampler object for invoking the sampler interface.

        Returns:
            A cirq.Sampler for running on the requested quantum_computer.
        """
        return RigettiQCSSampler(quantum_computer=self._quantum_computer, executor=self._executor, transformer=self._transformer)

    @staticmethod
    @_provide_default_client
    def list_quantum_processors(client: Optional[httpx.Client]) -> ListQuantumProcessorsResponse:
        """Retrieve a list of available Rigetti quantum processors.

        Args:
            client: Optional; A `httpx.Client` initialized with Rigetti QCS credentials
            and configuration. If not provided, `qcs_api_client` will initialize a
            configured client based on configured values in the current user's
            `~/.qcs` directory or default values.

        Returns:
            A qcs_api_client.models.ListQuantumProcessorsResponse containing the identifiers
            of the available quantum processors..
        """
        return cast(ListQuantumProcessorsResponse, list_quantum_processors(client=client).parsed)

    @staticmethod
    @_provide_default_client
    def get_quilt_calibrations(quantum_processor_id: str, client: Optional[httpx.Client]) -> GetQuiltCalibrationsResponse:
        """Retrieve the calibration data used for client-side Quil-T generation.

        Args:
            quantum_processor_id: The identifier of the Rigetti QCS quantum processor.
            client: Optional; A `httpx.Client` initialized with Rigetti QCS credentials
            and configuration. If not provided, `qcs_api_client` will initialize a
            configured client based on configured values in the current user's
            `~/.qcs` directory or default values.

        Returns:
            A qcs_api_client.models.GetQuiltCalibrationsResponse containing the
            device calibrations.
        """
        return cast(GetQuiltCalibrationsResponse, get_quilt_calibrations(client=client, quantum_processor_id=quantum_processor_id).parsed)

    @staticmethod
    @_provide_default_client
    def get_instruction_set_architecture(quantum_processor_id: str, client: Optional[httpx.Client]) -> InstructionSetArchitecture:
        """Retrieve the Instruction Set Architecture of a QuantumProcessor by ID. This
        includes site specific operations and native gate capabilities.

        Args:
            quantum_processor_id: The identifier of the Rigetti QCS quantum processor.
            client: Optional; A `httpx.Client` initialized with Rigetti QCS credentials
            and configuration. If not provided, `qcs_api_client` will initialize a
            configured client based on configured values in the current user's
            `~/.qcs` directory or default values.

        Returns:
            A qcs_api_client.models.InstructionSetArchitecture containing the device specification.
        """
        return cast(InstructionSetArchitecture, get_instruction_set_architecture(client=client, quantum_processor_id=quantum_processor_id).parsed)