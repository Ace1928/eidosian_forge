from __future__ import annotations
from enum import Enum
from typing import NamedTuple, Union, Optional, NewType, Any, List
import numpy as np
from qiskit import pulse
class WaveformChannel(pulse.channels.PulseChannel):
    """Dummy channel that doesn't belong to specific pulse channel."""
    prefix = 'w'

    def __init__(self):
        """Create new waveform channel."""
        super().__init__(0)