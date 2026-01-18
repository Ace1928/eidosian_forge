import numpy as np
from pygsp import utils
from . import Filter  # prevent circular import in Python < 3.5

            Evaluates Meyer function and scaling function

            * meyer wavelet kernel: supported on [2/3,8/3]
            * meyer scaling function kernel: supported on [0,4/3]
            