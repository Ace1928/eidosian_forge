import pytest
import cirq
def _with_measurement_key_mapping_(self, key_map):
    if not all((key in key_map for key in self._keys)):
        raise ValueError('missing keys')
    return MultiKeyGate([key_map[key] for key in self._keys])