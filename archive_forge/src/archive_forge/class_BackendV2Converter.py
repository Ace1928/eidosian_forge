from __future__ import annotations
import logging
import warnings
from typing import List, Iterable, Any, Dict, Optional
from qiskit.providers.backend import BackendV1, BackendV2
from qiskit.providers.backend import QubitProperties
from qiskit.providers.models.backendconfiguration import BackendConfiguration
from qiskit.providers.models.backendproperties import BackendProperties
from qiskit.circuit.controlflow import CONTROL_FLOW_OP_NAMES
from qiskit.providers.models.pulsedefaults import PulseDefaults
from qiskit.providers.options import Options
from qiskit.providers.exceptions import BackendPropertyError
class BackendV2Converter(BackendV2):
    """A converter class that takes a :class:`~.BackendV1` instance and wraps it in a
    :class:`~.BackendV2` interface.

    This class implements the :class:`~.BackendV2` interface and is used to enable
    common access patterns between :class:`~.BackendV1` and :class:`~.BackendV2`. This
    class should only be used if you need a :class:`~.BackendV2` and still need
    compatibility with :class:`~.BackendV1`.

    When using custom calibrations (or other custom workflows) it is **not** recommended
    to mutate the ``BackendV1`` object before applying this converter. For example, in order to
    convert a ``BackendV1`` object with a customized ``defaults().instruction_schedule_map``,
    which has a custom calibration for an operation, the operation name must be in
    ``configuration().basis_gates`` and ``name_mapping`` must be supplied for the operation.
    Otherwise, the operation will be dropped in the resulting ``BackendV2`` object.

    Instead it is typically better to add custom calibrations **after** applying this converter
    instead of updating ``BackendV1.defaults()`` in advance. For example::

        backend_v2 = BackendV2Converter(backend_v1)
        backend_v2.target.add_instruction(
            custom_gate, {(0, 1): InstructionProperties(calibration=custom_sched)}
        )
    """

    def __init__(self, backend: BackendV1, name_mapping: Optional[Dict[str, Any]]=None, add_delay: bool=True, filter_faulty: bool=True):
        """Initialize a BackendV2 converter instance based on a BackendV1 instance.

        Args:
            backend: The input :class:`~.BackendV1` based backend to wrap in a
                :class:`~.BackendV2` interface
            name_mapping: An optional dictionary that maps custom gate/operation names in
                ``backend`` to an :class:`~.Operation` object representing that
                gate/operation. By default most standard gates names are mapped to the
                standard gate object from :mod:`qiskit.circuit.library` this only needs
                to be specified if the input ``backend`` defines gates in names outside
                that set.
            add_delay: If set to true a :class:`~qiskit.circuit.Delay` operation
                will be added to the target as a supported operation for all
                qubits
            filter_faulty: If the :class:`~.BackendProperties` object (if present) for
                ``backend`` has any qubits or gates flagged as non-operational filter
                those from the output target.
        """
        self._backend = backend
        self._config = self._backend.configuration()
        super().__init__(provider=backend.provider, name=backend.name(), description=self._config.description, online_date=getattr(self._config, 'online_date', None), backend_version=self._config.backend_version)
        self._options = self._backend._options
        self._properties = None
        self._defaults = None
        if hasattr(self._backend, 'properties'):
            self._properties = self._backend.properties()
        if hasattr(self._backend, 'defaults'):
            self._defaults = self._backend.defaults()
        self._target = None
        self._name_mapping = name_mapping
        self._add_delay = add_delay
        self._filter_faulty = filter_faulty

    @property
    def target(self):
        """A :class:`qiskit.transpiler.Target` object for the backend.

        :rtype: Target
        """
        if self._target is None:
            self._target = convert_to_target(configuration=self._config, properties=self._properties, defaults=self._defaults, custom_name_mapping=self._name_mapping, add_delay=self._add_delay, filter_faulty=self._filter_faulty)
        return self._target

    @property
    def max_circuits(self):
        return self._config.max_experiments

    @classmethod
    def _default_options(cls):
        return Options()

    @property
    def dtm(self) -> float:
        return self._config.dtm

    @property
    def meas_map(self) -> List[List[int]]:
        return self._config.meas_map

    def drive_channel(self, qubit: int):
        return self._config.drive(qubit)

    def measure_channel(self, qubit: int):
        return self._config.measure(qubit)

    def acquire_channel(self, qubit: int):
        return self._config.acquire(qubit)

    def control_channel(self, qubits: Iterable[int]):
        return self._config.control(qubits)

    def run(self, run_input, **options):
        return self._backend.run(run_input, **options)