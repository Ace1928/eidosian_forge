import abc
import dataclasses
from typing import Any, Dict
import cirq
import cirq_google as cg
from cirq._compat import dataclass_repr
from cirq_google.engine.virtual_engine_factory import (
class SimulatedProcessorWithLocalDeviceRecord(SimulatedProcessorRecord):
    """A serializable record mapping a processor_id and optional noise spec to a
    completely local cg.AbstractProcessor

    Args:
        processor_id: The processor id we are emulating
        noise_strength: To do noisy simulation, set this to a positive float. The default
            of `0` will result in a noiseless state-vector simulation. If `float('inf')`
            is provided the simulator will be `cirq.ZerosSampler`. Otherwise, use
            a depolarizing model with this probability of noise.
    """

    def _get_input_device(self) -> 'cirq.Device':
        """Return a `cg.GridDevice` for the specified processor_id.

        Only 'rainbow' and 'weber' are recognized processor_ids and the device information
        may not be up-to-date, as it is completely local.
        """
        device_spec = _create_device_spec_from_template(MOST_RECENT_TEMPLATES[self.processor_id])
        device = cg.GridDevice.from_proto(device_spec)
        return device