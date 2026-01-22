import numpy as np
import sys
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
import pyaudio
class DynamicRangeCompressor:
    """Reduces the dynamic range of a sound."""

    def __init__(self, threshold: float, ratio: float):
        self.threshold = threshold
        self.ratio = ratio

    def apply_compression(self, sound: np.ndarray) -> np.ndarray:
        """Applies dynamic range compression to the sound."""
        pass