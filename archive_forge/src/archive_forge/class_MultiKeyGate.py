import pytest
import cirq
class MultiKeyGate:

    def __init__(self, keys):
        self._keys = frozenset((cirq.MeasurementKey.parse_serialized(key) for key in keys))

    def _measurement_key_names_(self):
        return frozenset((str(key) for key in self._keys))

    def _with_key_path_(self, path):
        return MultiKeyGate([str(key._with_key_path_(path)) for key in self._keys])