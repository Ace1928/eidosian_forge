import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
class my_cacw(clear_and_catch_warnings):
    class_modules = (sys.modules[__name__],)