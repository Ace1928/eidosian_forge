from typing import Sequence, Callable
from pennylane.transforms import transform
from pennylane.tape import QuantumTape
from pennylane import AmplitudeEmbedding
from pennylane._device import DeviceError
from pennylane.math import flatten, reshape
from pennylane.queuing import QueuingManager
A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        