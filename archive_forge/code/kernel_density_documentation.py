import numpy as np
from . import kernels
from ._kernel_base import GenericKDE, EstimatorSettings, gpke, \
Helper method to be able to pass needed vars to _compute_subset.