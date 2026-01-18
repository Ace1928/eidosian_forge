import sys
from PyQt5 import QtWidgets
from DAWModules import SoundModule
from PyQt5.QtCore import Qt
import pyaudio
import numpy as np
from typing import Dict, Type, List, Tuple
def update_module(self, module_name: str, value: int) -> None:
    """
        Updates module parameters based on GUI controls, specifically adjusting the 'volume' parameter as an example.

        Parameters:
            module_name (str): The name of the module to update.
            value (int): The new value from the slider, scaled to a range suitable for the module.
        """
    module = self.modules.get(module_name)
    if module:
        scaled_value = value / 100.0
        module.set_parameter('volume', scaled_value)