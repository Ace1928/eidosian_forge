import numpy as np
import sys
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
import pyaudio
class Equalizer:
    """Adjusts the balance between frequency components within a sound."""

    def apply_eq(self, sound: np.ndarray, frequency_bands: dict) -> np.ndarray:
        """Adjusts frequencies based on the provided frequency bands settings."""
        pass