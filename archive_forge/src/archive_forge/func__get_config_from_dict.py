import json
import os
from qiskit.exceptions import QiskitError
from qiskit.providers.models import BackendProperties, QasmBackendConfiguration
from .utils.json_decoder import (
from .fake_backend import FakeBackend
def _get_config_from_dict(self, conf):
    return QasmBackendConfiguration.from_dict(conf)