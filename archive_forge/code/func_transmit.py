import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from collections import deque
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox
from matplotlib.animation import FuncAnimation
import logging
import datetime
import sys
import cProfile
def transmit(self, signal, simulation_params):
    if not isinstance(signal, (int, float)):
        raise TypeError('Signal must be a number')
    if not isinstance(simulation_params, SimulationParameters):
        raise TypeError('Simulation parameters must be an instance of SimulationParameters')
    if self.invert_signal:
        signal = -signal
    repeated_signal = signal * self.strength
    self.delayed_signals.append(repeated_signal)
    return self.delayed_signals.popleft()