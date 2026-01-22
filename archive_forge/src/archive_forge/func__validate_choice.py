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
def _validate_choice(self, choice, name, valid_choices):
    if choice not in valid_choices:
        raise ValueError(f'{name} must be one of {valid_choices}')
    return choice