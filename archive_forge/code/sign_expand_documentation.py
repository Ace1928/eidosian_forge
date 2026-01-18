import json
from os import path
from typing import Sequence, Callable
import numpy as np
import pennylane as qml
from pennylane.transforms import transform
Projector on eigenspace of eigenvalue E_i