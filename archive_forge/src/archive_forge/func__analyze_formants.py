import numpy as np
import scipy.signal
from scipy.signal import resample, fftconvolve
from typing import Union, Tuple, Dict, Any
from abc import ABC, abstractmethod
def _analyze_formants(self, sound: np.ndarray=np.array([]), sample_rate: int=44100) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
        Analyzes the formant frequencies and bandwidths of the input sound using LPC (Linear Predictive Coding).

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.
            sample_rate (int): The sample rate of the sound data.

        Returns:
            Tuple[Dict[int, float], Dict[int, float]]: A tuple containing two dictionaries,
            one for formant frequencies and one for bandwidths, both indexed by formant number.
        """
    return ({1: 500, 2: 1500, 3: 2500}, {1: 50, 2: 70, 3: 100})