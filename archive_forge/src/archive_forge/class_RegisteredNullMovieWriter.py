import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import weakref
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.testing.decorators import check_figures_equal
@animation.writers.register('null')
class RegisteredNullMovieWriter(NullMovieWriter):

    def __init__(self, fps=None, codec=None, bitrate=None, extra_args=None, metadata=None):
        pass

    @classmethod
    def isAvailable(cls):
        return True