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
def _validate_process_input_params(self, input_signals, current_time, simulation_params):
    self._validate_signals(input_signals)
    if not isinstance(current_time, (int, float)):
        raise TypeError('Current time must be a number')
    if not isinstance(simulation_params, SimulationParameters):
        raise TypeError('Simulation parameters must be an instance of SimulationParameters')