from __future__ import annotations
from collections.abc import Callable
from collections import OrderedDict
from typing import Type
import logging
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import Backend
from qiskit.providers.provider import ProviderV1
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.providers.providerutils import filter_backends
from .basic_simulator import BasicSimulator
def _get_backend_instance(self, backend_cls: Type[Backend]) -> Backend:
    """
        Return an instance of a backend from its class.

        Args:
            backend_cls (class): backend class.
        Returns:
            Backend: a backend instance.
        Raises:
            QiskitError: if the backend could not be instantiated.
        """
    try:
        backend_instance = backend_cls(provider=self)
    except Exception as err:
        raise QiskitError(f'Backend {backend_cls} could not be instantiated: {err}') from err
    return backend_instance