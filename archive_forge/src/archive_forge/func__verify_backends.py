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
def _verify_backends(self) -> OrderedDict[str, Backend]:
    """
        Return the test backends in `BACKENDS` that are
        effectively available (as some of them might depend on the presence
        of an optional dependency or on the existence of a binary).

        Returns:
            dict[str:Backend]: a dict of test backend instances for
                the backends that could be instantiated, keyed by backend name.
        """
    ret = OrderedDict()
    for backend_cls in SIMULATORS:
        backend_instance = self._get_backend_instance(backend_cls)
        backend_name = backend_instance.name
        ret[backend_name] = backend_instance
    return ret