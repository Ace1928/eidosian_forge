from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Tuple
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
Run the MinimumPoint pass on `dag`.