import concurrent
import os
import time
import uuid
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass
from typing import (
import numpy as np
import pandas as pd
import tqdm
from cirq import ops, devices, value, protocols
from cirq.circuits import Circuit, Moment
from cirq.experiments.random_quantum_circuit_generation import CircuitLibraryCombination
This closure will execute a list of `tasks` with one call to
        `run_batch` on the provided sampler for a given number of repetitions.

        It also keeps a record of the circuit library combinations in order to
        back out which qubit pairs correspond to each pair index. We tag
        our return value with this so it is in the resultant DataFrame, which
        is very convenient for dealing with the results (but not strictly
        necessary, as the information could be extracted from (`layer_i`, `pair_i`).
        