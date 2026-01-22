import numpy as np
import scipy.signal
from scipy.signal import resample, fftconvolve
from typing import Union, Tuple, Dict, Any
from abc import ABC, abstractmethod
def _generate_impulse_response(self, length: int) -> np.ndarray:
    """
        Generates an impulse response using an exponentially decaying noise signal, providing the basis for the reverb effect.

        Parameters:
            length (int): The length of the impulse response, determined by the input sound length.

        Returns:
            np.ndarray: The generated impulse response, an array of exponentially decaying values modulated by random noise.
        """
    t = np.linspace(0, self.decay, length)
    impulse_response = np.exp(-t / self.decay) * np.random.randn(length)
    return impulse_response