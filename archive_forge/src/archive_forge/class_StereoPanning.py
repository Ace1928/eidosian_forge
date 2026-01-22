import numpy as np
import sys
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
import pyaudio
class StereoPanning:
    """Manages the distribution of a sound's signal across a stereo field."""

    def pan_stereo(self, sound: np.ndarray, pan: float) -> np.ndarray:
        """Pans sound between left and right channels based on pan parameter."""
        pass