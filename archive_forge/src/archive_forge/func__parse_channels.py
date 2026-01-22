import warnings
import collections
import json
import os
import re
from typing import List, Iterable
from qiskit import circuit
from qiskit.providers.models import BackendProperties, BackendConfiguration, PulseDefaults
from qiskit.providers import BackendV2, BackendV1
from qiskit import pulse
from qiskit.exceptions import QiskitError
from qiskit.utils import optionals as _optionals
from qiskit.providers import basic_provider
from qiskit.transpiler import Target
from qiskit.providers.backend_compat import convert_to_target
from .utils.json_decoder import (
def _parse_channels(self, channels):
    type_map = {'acquire': pulse.AcquireChannel, 'drive': pulse.DriveChannel, 'measure': pulse.MeasureChannel, 'control': pulse.ControlChannel}
    identifier_pattern = re.compile('\\D+(?P<index>\\d+)')
    channels_map = {'acquire': collections.defaultdict(list), 'drive': collections.defaultdict(list), 'measure': collections.defaultdict(list), 'control': collections.defaultdict(list)}
    for identifier, spec in channels.items():
        channel_type = spec['type']
        out = re.match(identifier_pattern, identifier)
        if out is None:
            continue
        channel_index = int(out.groupdict()['index'])
        qubit_index = tuple(spec['operates']['qubits'])
        chan_obj = type_map[channel_type](channel_index)
        channels_map[channel_type][qubit_index].append(chan_obj)
    setattr(self, 'channels_map', channels_map)