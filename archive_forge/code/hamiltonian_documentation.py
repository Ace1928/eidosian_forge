import itertools
import numbers
from collections.abc import Iterable
from copy import copy
import functools
from typing import List
import numpy as np
import scipy
import pennylane as qml
from pennylane.operation import Observable, Tensor
from pennylane.wires import Wires
Function that formats the wires.