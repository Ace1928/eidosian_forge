import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
class mod:

    def __init__(self):
        self.__warningregistry__ = {'warning1': 1, 'warning2': 2}