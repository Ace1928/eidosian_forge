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
def transform_array_input(array_input, signal_size, num_neurons):
    return np.reshape(array_input, (num_neurons, signal_size))