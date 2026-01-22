from typing import (
import re
import warnings
from dataclasses import dataclass
import cirq
from cirq_google import ops
from cirq_google import transformers
from cirq_google.api import v2
from cirq_google.devices import known_devices
from cirq_google.experimental import ops as experimental_ops
@classmethod
def _from_device_information(cls, *, qubit_pairs: Collection[Tuple[cirq.GridQubit, cirq.GridQubit]], gateset: cirq.Gateset, gate_durations: Optional[Mapping[cirq.GateFamily, cirq.Duration]]=None, all_qubits: Optional[Collection[cirq.GridQubit]]=None) -> 'GridDevice':
    """Constructs a GridDevice using the device information provided.

        EXPERIMENTAL: this method may have changes which are not backward compatible in the future.

        This is a convenience method for constructing a GridDevice given partial gateset and
        gate_duration information: for every distinct gate, only one representation needs to be in
        gateset and gate_duration. The remaining representations will be automatically generated.

        For example, if the input gateset contains only `cirq.PhasedXZGate`, and the input
        gate_durations is `{cirq.GateFamily(cirq.PhasedXZGate): cirq.Duration(picos=3)}`,
        `GridDevice.metadata.gateset` will be

        ```
        cirq.Gateset(cirq.PhasedXZGate, cirq.XPowGate, cirq.YPowGate, cirq.PhasedXPowGate)
        ```

        and `GridDevice.metadata.gate_durations` will be

        ```
        {
            cirq.GateFamily(cirq.PhasedXZGate): cirq.Duration(picos=3),
            cirq.GateFamily(cirq.XPowGate): cirq.Duration(picos=3),
            cirq.GateFamily(cirq.YPowGate): cirq.Duration(picos=3),
            cirq.GateFamily(cirq.PhasedXPowGate): cirq.Duration(picos=3),
        }
        ```

        This method reduces the complexity of constructing `GridDevice` on server side by requiring
        only the bare essential device information.

        Args:
            qubit_pairs: Collection of bidirectional qubit couplings available on the device.
            gateset: The gate set supported by the device.
            gate_durations: Optional mapping from gates supported by the device to their timing
                estimates. Not every gate is required to have an associated duration.
            out: If set, device information will be serialized into this DeviceSpecification.

        Raises:
            ValueError: If a pair contains two identical qubits.
            ValueError: If `gateset` contains invalid GridDevice gates.
            ValueError: If `gate_durations` contains keys which are not in `gateset`.
            ValueError: If multiple gate families in gate_durations can
                represent a particular gate, but they have different durations.
            ValueError: If all_qubits is provided and is not a superset
                of all the qubits found in qubit_pairs.
        """
    metadata = cirq.GridDeviceMetadata(qubit_pairs=qubit_pairs, gateset=gateset, gate_durations=gate_durations, all_qubits=all_qubits)
    incomplete_device = GridDevice(metadata)
    return GridDevice.from_proto(incomplete_device.to_proto())