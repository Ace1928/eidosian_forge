from inspect import signature
import numpy as np
from scipy.optimize import brute, shgo
import pennylane as qml
def _univariate_fn(x):
    return fn(*args[:arg_idx], the_arg + shift_vec * x, *args[arg_idx + 1:], **kwargs)