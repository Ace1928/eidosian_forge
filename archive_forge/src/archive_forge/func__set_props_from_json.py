import json
import os
from qiskit.exceptions import QiskitError
from qiskit.providers.models import BackendProperties, QasmBackendConfiguration
from .utils.json_decoder import (
from .fake_backend import FakeBackend
def _set_props_from_json(self):
    if not self.props_filename:
        raise QiskitError('No properties file has been defined')
    props = self._load_json(self.props_filename)
    decode_backend_properties(props)
    self._properties = BackendProperties.from_dict(props)