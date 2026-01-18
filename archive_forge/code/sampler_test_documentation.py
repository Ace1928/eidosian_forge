from typing import Sequence
import pytest
import duet
import numpy as np
import pandas as pd
import sympy
import cirq
A simple, deterministic mock sampler.
        Pretends to sample from a state vector with a 3:1 balance between the
        probabilities of the |0) and |1) state.
        