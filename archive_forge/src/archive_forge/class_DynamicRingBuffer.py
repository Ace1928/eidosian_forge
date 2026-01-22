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
class DynamicRingBuffer:

    def __init__(self, initial_size=1000):
        self.data = deque(maxlen=initial_size)

    def append(self, item):
        self.data.append(item)

    def get(self):
        return list(self.data)