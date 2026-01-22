from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
Return the requested collection of test cases, as an array of dicts with subset-specific keys

    Allowed values of collection:
    'original': The original benchmarking functions.
         Real-valued functions of real-valued inputs on an interval with a zero.
         f1, .., f3 are continuous and infinitely differentiable
         f4 has a single discontinuity at the root
         f5 has a root at 1 replacing a 1st order pole
         f6 is randomly positive on one side of the root, randomly negative on the other
    'aps': The test problems in the TOMS "Algorithm 748: Enclosing Zeros of Continuous Functions"
         paper by Alefeld, Potra and Shi. Real-valued functions of
         real-valued inputs on an interval with a zero.
         Suitable for methods which start with an enclosing interval, and
         derivatives up to 2nd order.
    'complex': Some complex-valued functions of complex-valued inputs.
         No enclosing bracket is provided.
         Suitable for methods which use one or more starting values, and
         derivatives up to 2nd order.

    The dictionary keys will be a subset of
    ["f", "fprime", "fprime2", "args", "bracket", "a", b", "smoothness", "x0", "x1", "root", "ID"]
    