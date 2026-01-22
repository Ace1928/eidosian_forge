import numpy as np
import sys
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
import pyaudio
def amplitude_modulation(self, carrier: np.ndarray, modulator: np.ndarray, index: float) -> np.ndarray:
    """Performs amplitude modulation on a sound."""
    return carrier * (1 + index * modulator)