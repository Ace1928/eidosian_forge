from typing import Callable, Optional, Sequence, Union
import cirq
Wrapper around `cirq.Sampler` that performs device validation.

        This sampler will delegate to the wrapping sampler after
        performing validation on the circuit(s) given to the sampler.

        Args:
            device: `cirq.Device` that will validate_circuit before sampling.
            validator: A callable that will do any additional validation
               beyond the device.  For instance, this can perform serialization
               checks.  Note that this function takes a list of circuits and
               sweeps so that batch functionality can also be tested.
            sampler: sampler wrapped by this object.  After validating,
                samples will be returned by this enclosed `cirq.Sampler`.
        