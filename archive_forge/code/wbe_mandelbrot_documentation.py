import logging
import math
import os
import sys
from taskflow import engines
from taskflow.engines.worker_based import worker
from taskflow.patterns import unordered_flow as uf
from taskflow import task
from taskflow.utils import threading_utils
Returns the number of iterations before the computation "escapes".

        Given the real and imaginary parts of a complex number, determine if it
        is a candidate for membership in the mandelbrot set given a fixed
        number of iterations.
        