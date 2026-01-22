from copy import copy
from typing import Tuple
import numpy as np
import numpy.linalg as npl
import pennylane as qml
from pennylane.operation import Operation, Operator
from pennylane.wires import Wires
from pennylane import math
Returns a triplet of angles representing the single-qubit decomposition
        of the matrix of the target operation using ZYZ rotations.
        